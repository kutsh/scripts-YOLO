#!/usr/bin/env python3
"""
Merge near-duplicate / fragmented YOLO labels of the same class.

Handles both YOLO label formats, and files that mix them (coordinates
normalized to 0-1):
  detection     class cx cy w h [conf]
  segmentation  class x1 y1 x2 y2 x3 y3 ... [conf]      (polygon, 3+ points)

Instances of the SAME class that are extremely close are merged into one.
Grouping is transitive: if A is close to B and B is close to C, all three
become one instance.

Two instances are "close" if ANY enabled criterion is met:
  --iou T           overlap IoU >= T                          (default 0.5)
  --gap D           shortest distance between outlines <= D   (off by default)
                    0 means touching or overlapping; 0.01 means within 1% of
                    the image
  --center-dist D   distance between centers <= D             (off by default)
For polygons these are measured on the actual shapes, not their bounding boxes.

Distances are in normalized units. With --img-size W H, --gap and
--center-dist are in pixels instead (all images must share that size).

How a group is merged:
  boxes     --mode union (default): the smallest box enclosing the group
            --mode mean: area-weighted average box
  polygons  one outline covering the union of the shapes. Gaps up to --gap are
            bridged; pieces still apart (grouped only via --center-dist) are
            joined by their convex hull. Holes are dropped, since YOLO polygons
            can't represent them. A box grouped with polygons counts as a
            4-point polygon.
Instances that aren't merged are written back unchanged.

Polygon labels need numpy and OpenCV (both come with ultralytics; otherwise
`pip install opencv-python-headless`). Polygon overlap and merged outlines are
computed on a pixel grid: the image's resolution when --img-size is given or
the image is found (see --images), otherwise --raster-size.

Preview images (needs Pillow: pip install pillow):
  --preview DIR  saves a side-by-side before/after image for each changed file.
                 Instances involved in a merge are drawn thick and tagged "xN".
                 Images are looked up in the sibling `images/` folder of the
                 standard YOLO layout (…/labels/x.txt -> …/images/x.jpg), next
                 to the label file, or in --images DIR. If none is found, the
                 labels are drawn on a blank canvas.

Examples:
  python merge_yolo_boxes.py labels/ -o labels_merged/
  python merge_yolo_boxes.py labels/ -o out/ --iou 0.3 --gap 0.005
  python merge_yolo_boxes.py labels/ -o out/ --gap 5 --img-size 1920 1080
  python merge_yolo_boxes.py labels/ --in-place --dry-run
  python merge_yolo_boxes.py labels/ --dry-run --gap 0.002 --preview previews/
"""

import argparse
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ------------------------------------------------------------ box geometry ---

def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def to_cxcywh(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def edge_gap(a, b, sx=1.0, sy=1.0):
    """Shortest distance between the two rectangles (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2])) * sx
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3])) * sy
    return math.hypot(dx, dy)


def center_dist(a, b, sx=1.0, sy=1.0):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot((ax - bx) * sx, (ay - by) * sy)


class Obj:
    """One label line: an axis-aligned box or a polygon, normalized coords."""
    __slots__ = ("cls", "kind", "pts", "bbox", "conf", "_arr")

    def __init__(self, cls, kind, pts, conf=None):
        self.cls, self.kind, self.pts, self.conf = cls, kind, pts, conf
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        self._arr = None

    @classmethod
    def box(klass, cls, xyxy, conf=None):
        x1, y1, x2, y2 = xyxy
        return klass(cls, "box", [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], conf)


# -------------------------------------------------------- polygon geometry ---

def need_cv():
    try:
        import numpy as np
        import cv2
    except ImportError:
        sys.exit("error: segmentation (polygon) labels need numpy and OpenCV:\n"
                 "  pip install opencv-python-headless")
    return np, cv2


def as_array(o, scale):
    """Polygon vertices as an (n, 2) array in distance units (cached)."""
    if o._arr is None:
        np, _ = need_cv()
        o._arr = np.asarray(o.pts, dtype=float) * np.asarray(scale, dtype=float)
    return o._arr


_CHUNK = 1_000_000  # max (vertex x edge) pairs evaluated at once


def _chunks(P, n_edges):
    step = max(1, _CHUNK // max(1, n_edges))
    for i in range(0, len(P), step):
        yield i, P[i:i + step]


def _inside(P, V):
    """Even-odd test: which points of P (n,2) lie inside polygon V (m,2)."""
    np, _ = need_cv()
    W = np.roll(V, -1, axis=0)
    x1, y1, x2, y2 = V[:, 0], V[:, 1], W[:, 0], W[:, 1]
    out = []
    for _, c in _chunks(P, len(V)):
        x, y = c[:, :1], c[:, 1:]
        spans = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        out.append((spans & (x < xs)).sum(axis=1) % 2 == 1)
    return np.concatenate(out)


def _edges_cross(A, B):
    """True if any edge of A properly crosses any edge of B."""
    np, _ = need_cv()
    A2, B2 = np.roll(A, -1, axis=0), np.roll(B, -1, axis=0)
    b1, b2 = B[None], B2[None]

    def orient(p, q, r):
        return ((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))

    for i, c in _chunks(A, len(B)):
        a1, a2 = c[:, None], A2[i:i + len(c), None]
        if ((orient(b1, b2, a1) * orient(b1, b2, a2) < 0)
                & (orient(a1, a2, b1) * orient(a1, a2, b2) < 0)).any():
            return True
    return False


def _vertex_edge_dist(P, V):
    """Smallest distance from any point of P to any edge of polygon V."""
    np, _ = need_cv()
    A = V
    AB = np.roll(V, -1, axis=0) - A
    L2 = (AB ** 2).sum(axis=1)
    L2[L2 == 0] = 1.0
    best = math.inf
    for _, c in _chunks(P, len(V)):
        AP = c[:, None, :] - A[None]
        t = np.clip((AP * AB[None]).sum(axis=-1) / L2, 0.0, 1.0)
        d = AP - t[..., None] * AB[None]
        best = min(best, float(np.sqrt((d ** 2).sum(axis=-1)).min()))
    return best


def poly_distance(A, B):
    """Exact shortest distance between two polygons (0 if they touch/overlap)."""
    if _inside(A, B).any() or _inside(B, A).any() or _edges_cross(A, B):
        return 0.0
    return min(_vertex_edge_dist(A, B), _vertex_edge_dist(B, A))


def poly_centroid(V):
    np, _ = need_cv()
    x, y = V[:, 0], V[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    c = x * y2 - x2 * y
    a = c.sum() / 2
    if abs(a) < 1e-12:
        return V.mean(axis=0)
    return np.array([((x + x2) * c).sum(), ((y + y2) * c).sum()]) / (6 * a)


def rasterize(objs, grid, pad=1):
    """Fill each object into its own mask over their shared region of a
    (Wr, Hr) pixel grid. Returns (masks, (x0, y0) offset of the region)."""
    np, cv2 = need_cv()
    Wr, Hr = grid
    x0 = math.floor(min(o.bbox[0] for o in objs) * Wr) - pad
    y0 = math.floor(min(o.bbox[1] for o in objs) * Hr) - pad
    x1 = math.ceil(max(o.bbox[2] for o in objs) * Wr) + pad
    y1 = math.ceil(max(o.bbox[3] for o in objs) * Hr) + pad
    masks = []
    for o in objs:
        m = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
        pts = np.round(np.asarray(o.pts) * (Wr, Hr) - (x0, y0)).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        masks.append(m)
    return masks, (x0, y0)


def poly_iou(a, b, grid):
    np, _ = need_cv()
    ma, mb = rasterize([a, b], grid)[0]
    union = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / union if union else 0.0


def merge_polygons(members, cfg, grid):
    """Outline of the union of the members' shapes, as normalized points."""
    np, cv2 = need_cv()
    Wr, Hr = grid
    sx, sy = cfg.scale
    gap_px = (cfg.gap or 0) * max(Wr / sx, Hr / sy)   # --gap in grid pixels
    r = math.ceil(gap_px / 2) + 1                      # closing radius
    masks, (x0, y0) = rasterize(members, grid, pad=r + 2)
    mask = np.bitwise_or.reduce(masks)

    def n_parts(m):
        return cv2.connectedComponents(m, connectivity=8)[0] - 1

    if n_parts(mask) > 1:
        # bridge the small gaps between pieces (morphological closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        if n_parts(closed) == 1:
            mask = closed
        else:  # pieces too far apart to bridge: join them by their convex hull
            pts = np.concatenate([np.asarray(m.pts) for m in members])
            pts = np.round(pts * (Wr, Hr) - (x0, y0)).astype(np.int32)
            cv2.fillPoly(mask, [cv2.convexHull(pts)], 1)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.5, True)
    outline = outline.reshape(-1, 2)
    if len(outline) < 3:  # degenerate (zero-area) input: fall back to a box
        return Obj.box(None, (min(m.bbox[0] for m in members),
                              min(m.bbox[1] for m in members),
                              max(m.bbox[2] for m in members),
                              max(m.bbox[3] for m in members))).pts
    return [((x + x0) / Wr, (y + y0) / Hr) for x, y in outline.tolist()]


# -------------------------------------------------------------- clustering ---

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[rj] = ri


def are_close(a, b, cfg, grid):
    sx, sy = cfg.scale
    if a.kind == "box" and b.kind == "box":
        A, B = a.bbox, b.bbox
        if cfg.iou is not None and iou(A, B) >= cfg.iou:
            return True
        if cfg.gap is not None and edge_gap(A, B, sx, sy) <= cfg.gap:
            return True
        if cfg.center_dist is not None and center_dist(A, B, sx, sy) <= cfg.center_dist:
            return True
        return False

    # polygons can't be closer than their bounding boxes: cheap rejection first
    bbox_gap = edge_gap(a.bbox, b.bbox, sx, sy)
    if bbox_gap > cfg.reach:
        return False
    A, B = as_array(a, cfg.scale), as_array(b, cfg.scale)
    if cfg.iou is not None and bbox_gap == 0 and poly_iou(a, b, grid()) >= cfg.iou:
        return True
    if cfg.gap is not None and poly_distance(A, B) <= cfg.gap:
        return True
    if cfg.center_dist is not None:
        dx, dy = poly_centroid(A) - poly_centroid(B)
        if math.hypot(dx, dy) <= cfg.center_dist:
            return True
    return False


def merge_group(members, cfg, grid):
    if len(members) == 1:
        return members[0]
    cls = members[0].cls
    confs = [m.conf for m in members]
    conf = max(confs) if all(c is not None for c in confs) else None
    if all(m.kind == "box" for m in members):
        boxes = [m.bbox for m in members]
        if cfg.mode == "union":
            merged = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
        else:  # mean, weighted by area
            weights = [area(b) or 1e-12 for b in boxes]
            tot = sum(weights)
            merged = tuple(sum(b[k] * w for b, w in zip(boxes, weights)) / tot
                           for k in range(4))
        return Obj.box(cls, merged, conf)
    return Obj(cls, "poly", merge_polygons(members, cfg, grid()), conf)


def merge_objects(objects, cfg, grid):
    """Returns ([(merged_obj, member_objs), ...], n_removed)."""
    by_class = defaultdict(list)
    for o in objects:
        by_class[o.cls].append(o)

    out, removed = [], 0
    for objs in by_class.values():
        n = len(objs)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if uf.find(i) != uf.find(j) and are_close(objs[i], objs[j], cfg, grid):
                    uf.union(i, j)

        groups = defaultdict(list)
        for i in range(n):
            groups[uf.find(i)].append(i)

        for idxs in sorted(groups.values(), key=lambda g: g[0]):
            members = [objs[i] for i in idxs]
            out.append((merge_group(members, cfg, grid), members))
            removed += len(idxs) - 1
    return out, removed


# --------------------------------------------------------------------- I/O ---

def parse_line(parts, fmt):
    """Returns an Obj, or None if the line isn't a label in the given format."""
    n = len(parts)
    vals = [float(v) for v in parts[1:]]  # ValueError -> not a label
    cls = parts[0]
    if fmt != "segment" and n in (5, 6):
        conf = vals[4] if n == 6 else None
        return Obj.box(cls, to_xyxy(*vals[:4]), conf)
    if fmt != "detect" and n >= 7:
        # odd count: plain polygon; even count: polygon + confidence
        coords, conf = (vals, None) if n % 2 == 1 else (vals[:-1], vals[-1])
        return Obj(cls, "poly", list(zip(coords[0::2], coords[1::2])), conf)
    return None


def parse_file(path, fmt):
    """Returns (objects, passthrough_lines). Lines that aren't labels are kept
    untouched and reported."""
    objects, passthrough = [], []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            obj = parse_line(parts, fmt)
        except ValueError:
            obj = None
        if obj is None:
            print(f"  warning: {path}:{lineno} is not a YOLO {fmt} label "
                  f"({len(parts)} values), kept as-is", file=sys.stderr)
            passthrough.append(line)
        else:
            objects.append(obj)
    return objects, passthrough


def format_obj(o, clip):
    c = (lambda v: min(1.0, max(0.0, v))) if clip else (lambda v: v)
    if o.kind == "box":
        vals = to_cxcywh(*(c(v) for v in o.bbox))
    else:
        vals = [c(v) for p in o.pts for v in p]
    s = o.cls + "".join(f" {v:.6f}" for v in vals)
    if o.conf is not None:
        s += f" {o.conf:.6f}"
    return s


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def find_image(label_path, base, images_dir):
    """Look for the image that goes with a label file."""
    stems = []
    if images_dir:
        stems.append(images_dir / label_path.relative_to(base).with_suffix(""))
    else:
        parts = label_path.with_suffix("").parts
        for i in range(len(parts) - 1, -1, -1):  # …/labels/… -> …/images/…
            if parts[i] == "labels":
                stems.append(Path(*parts[:i], "images", *parts[i + 1:]))
                break
        stems.append(label_path.with_suffix(""))  # image beside the label
    for stem in stems:
        for ext in IMG_EXTS:
            for e in (ext, ext.upper()):
                p = stem.with_name(stem.name + e)
                if p.exists():
                    return p
    return None


MIN_GRID = 2048  # small images are upsampled by an integer factor up to this


def image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            if im.getexif().get(0x0112) in (5, 6, 7, 8):  # EXIF rotated 90°
                w, h = h, w
            return w, h
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            return img.shape[1], img.shape[0]
    except Exception:
        pass
    return None


def grid_for(label_path, base, cfg):
    """Pixel grid used to rasterize polygons for this label file."""
    size = tuple(cfg.img_size) if cfg.img_size else None
    if size is None:
        img = find_image(label_path, base, cfg.images)
        size = image_size(img) if img else None
    if size is None:
        return (cfg.raster_size, cfg.raster_size)
    f = max(1, math.ceil(MIN_GRID / max(size)))
    return (size[0] * f, size[1] * f)


# ----------------------------------------------------------------- preview ---

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
           (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40),
           (255, 225, 25), (128, 0, 0), (170, 255, 195), (128, 128, 0),
           (255, 215, 180), (0, 0, 128)]


def class_color(cls):
    try:
        i = int(cls)
    except ValueError:
        i = sum(map(ord, cls))
    return PALETTE[i % len(PALETTE)]


def load_font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render_preview(before, after, image_path, out_path, cfg, names):
    from PIL import Image, ImageDraw, ImageOps

    if image_path:
        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    else:
        w, h = cfg.img_size or (640, 640)
        img = Image.new("RGB", (w, h), (45, 45, 48))
    scale = min(1.0, cfg.preview_max / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    W, H = img.size

    thick = max(2, round(max(W, H) / 250))
    thin = max(1, thick // 2)
    font = load_font(max(11, round(max(W, H) / 60)))
    show_all_tags = len(before) <= 40

    def px_pts(o):
        return [(x * W, y * H) for x, y in o.pts]

    def px_box(o):
        b = o.bbox
        return (b[0] * W, b[1] * H, b[2] * W, b[3] * H)

    def name(cls):
        return names.get(cls, cls)

    def tag(draw, box, text, color, placed):
        l, t, r, bt = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l + 6, bt - t + 4
        x = min(max(0, box[0]), W - tw)
        y = box[1] - th if box[1] - th >= 0 else box[1]  # above the box if it fits
        # slide down past any tag already drawn in this panel
        for _ in range(len(placed) + 1):
            hit = next((p for p in placed if x < p[2] and x + tw > p[0]
                        and y < p[3] and y + th > p[1]), None)
            if hit is None:
                break
            y = hit[3]
        placed.append((x, y, x + tw, y + th))
        draw.rectangle([x, y, x + tw, y + th], fill=color)
        draw.text((x + 3 - l, y + 2 - t), text, fill=(0, 0, 0), font=font)

    def draw_panel(items):
        """items: (obj, emphasized, tag text or None)."""
        panel = img.convert("RGBA")
        overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for o, strong, _ in items:  # translucent fill for polygons
            if o.kind == "poly" and len(o.pts) >= 3:
                od.polygon(px_pts(o), fill=class_color(o.cls) + (80 if strong else 45,))
        panel = Image.alpha_composite(panel, overlay).convert("RGB")
        d = ImageDraw.Draw(panel)
        for o, strong, _ in items:  # outlines, then tags on top
            width = thick if strong else thin
            if o.kind == "box":
                d.rectangle(px_box(o), outline=class_color(o.cls), width=width)
            else:
                pts = px_pts(o)
                d.line(pts + [pts[0]], fill=class_color(o.cls), width=width, joint="curve")
        placed = []
        for o, _, text in items:
            if text:
                tag(d, px_box(o), text, class_color(o.cls), placed)
        return panel

    left = draw_panel([(m, len(members) > 1, name(m.cls) if show_all_tags else None)
                       for _, members in after for m in members])
    right = draw_panel([(o, len(members) > 1,
                         f"{name(o.cls)} x{len(members)}" if len(members) > 1
                         else (name(o.cls) if show_all_tags else None))
                        for o, members in after])

    # assemble with a header over each panel
    n_before, n_after = len(before), len(after)
    header_font = load_font(max(14, round(W / 40)))
    header_h = round(max(14, W / 40) * 2)
    gap = 12
    canvas = Image.new("RGB", (W * 2 + gap, H + header_h), (255, 255, 255))
    canvas.paste(left, (0, header_h))
    canvas.paste(right, (W + gap, header_h))
    d = ImageDraw.Draw(canvas)
    src = image_path.name if image_path else "no image found, blank canvas"
    d.text((8, header_h / 2), f"Before: {n_before} labels  ({src})",
           fill=(20, 20, 20), font=header_font, anchor="lm")
    d.text((W + gap + 8, header_h / 2),
           f"After: {n_after} labels  ({n_before - n_after} merged away)",
           fill=(20, 20, 20), font=header_font, anchor="lm")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


# -------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path,
                    help="labels directory (searched recursively) or a single .txt file")
    dest = ap.add_mutually_exclusive_group()
    dest.add_argument("-o", "--output", type=Path,
                      help="output directory (mirrors the input structure)")
    dest.add_argument("--in-place", action="store_true",
                      help="overwrite the input files")
    ap.add_argument("--format", choices=["auto", "detect", "segment"], default="auto",
                    help="label format: auto (default) reads 5-6 values as a box "
                         "and 7+ as a polygon")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="merge if IoU >= this (default 0.5; negative disables)")
    ap.add_argument("--gap", type=float, default=None,
                    help="merge if the gap between outlines is <= this")
    ap.add_argument("--center-dist", type=float, default=None,
                    help="merge if center distance <= this")
    ap.add_argument("--img-size", type=int, nargs=2, metavar=("W", "H"),
                    help="interpret --gap / --center-dist in pixels for this image size")
    ap.add_argument("--mode", choices=["union", "mean"], default="union",
                    help="how to combine a group of boxes (default union)")
    ap.add_argument("--raster-size", type=int, default=4096, metavar="PX",
                    help="pixel grid for polygon merging when the image size is "
                         "unknown (default 4096)")
    ap.add_argument("--no-clip", action="store_true",
                    help="don't clip merged coordinates to the [0,1] image bounds")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be merged without writing labels "
                         "(previews are still written)")
    pv = ap.add_argument_group("preview")
    pv.add_argument("--preview", type=Path, metavar="DIR",
                    help="write before/after preview images to DIR")
    pv.add_argument("--images", type=Path, metavar="DIR",
                    help="image folder mirroring the labels folder "
                         "(default: auto-detect …/images/ or beside the label)")
    pv.add_argument("--preview-all", action="store_true",
                    help="also preview files where nothing was merged")
    pv.add_argument("--preview-limit", type=int, default=100, metavar="N",
                    help="stop after N previews (default 100; 0 = no limit)")
    pv.add_argument("--preview-max", type=int, default=1280, metavar="PX",
                    help="downscale each panel so its longest side is at most PX "
                         "(default 1280)")
    cfg = ap.parse_args()

    if cfg.preview:
        try:
            import PIL  # noqa: F401
        except ImportError:
            ap.error("--preview needs Pillow: pip install pillow")
    if cfg.iou is not None and cfg.iou < 0:
        cfg.iou = None
    if cfg.iou is None and cfg.gap is None and cfg.center_dist is None:
        ap.error("all merge criteria are disabled; set --iou, --gap or --center-dist")
    if not cfg.dry_run and not cfg.output and not cfg.in_place:
        ap.error("choose --output DIR, --in-place, or --dry-run")
    cfg.scale = tuple(cfg.img_size) if cfg.img_size else (1.0, 1.0)
    cfg.reach = max(cfg.gap or 0.0, cfg.center_dist or 0.0)

    root = cfg.input
    if root.is_file():
        files, base = [root], root.parent
    elif root.is_dir():
        files, base = sorted(root.rglob("*.txt")), root
    else:
        ap.error(f"{root} does not exist")
    files = [f for f in files if f.name != "classes.txt"]

    if cfg.output and not cfg.dry_run and cfg.output.resolve() == base.resolve():
        ap.error("output directory equals input; use --in-place to overwrite")

    names = {}
    if (base / "classes.txt").exists():
        names = {str(i): n.strip() for i, n in
                 enumerate((base / "classes.txt").read_text().splitlines())}

    total_before = total_after = files_changed = 0
    n_previews = n_blank = 0
    for f in files:
        objects, passthrough = parse_file(f, cfg.format)

        cache = {}

        def grid(f=f, cache=cache):  # resolved only if polygons need it
            if "g" not in cache:
                cache["g"] = grid_for(f, base, cfg)
            return cache["g"]

        merged, removed = merge_objects(objects, cfg, grid)
        total_before += len(objects)
        total_after += len(merged)
        if removed:
            files_changed += 1
            print(f"{f.relative_to(base)}: {len(objects)} -> {len(merged)} labels")

        if (cfg.preview and (removed or cfg.preview_all)
                and (cfg.preview_limit <= 0 or n_previews < cfg.preview_limit)):
            image_path = find_image(f, base, cfg.images)
            n_blank += image_path is None
            out = cfg.preview / f.relative_to(base).with_suffix(".jpg")
            render_preview(objects, merged, image_path, out, cfg, names)
            n_previews += 1

        if cfg.dry_run:
            continue
        target = f if cfg.in_place else cfg.output / f.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        if removed:
            lines = [format_obj(o, not cfg.no_clip) for o, _ in merged]
            target.write_text("\n".join(lines + passthrough) + "\n")
        elif not cfg.in_place:
            shutil.copy2(f, target)  # untouched files are copied byte-for-byte

    # carry classes.txt over, if present
    if cfg.output and not cfg.dry_run and (base / "classes.txt").exists():
        cfg.output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "classes.txt", cfg.output / "classes.txt")

    print(f"\n{len(files)} files scanned, {files_changed} changed; "
          f"{total_before} labels -> {total_after} "
          f"({total_before - total_after} merged away)"
          + ("  [dry run, nothing written]" if cfg.dry_run else ""))
    if cfg.preview:
        msg = f"{n_previews} preview image(s) written to {cfg.preview}"
        if n_blank:
            msg += f" ({n_blank} without a matching image, drawn on a blank canvas)"
        if cfg.preview_limit > 0 and n_previews >= cfg.preview_limit:
            msg += f"; limit of {cfg.preview_limit} reached, use --preview-limit 0 for all"
        print(msg)


if __name__ == "__main__":
    main()
