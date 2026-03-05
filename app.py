from __future__ import annotations
import random
import pygame
import sys

# ── Config ────────────────────────────────────────────────────────────────────

SEED        = 4
MAP_W       = 32
MAP_H       = 32
MAX_HEIGHT  = 6
HIGH_ALT    = 5
LOW_ALT     = 1
MAX_SLOPE   = 1

# Base sprite dimensions (native spritesheet size — never changes)
TILE_W_BASE = 64
TILE_H_BASE = 64
TILE_Z_BASE = 32

# Scale factor — 0.5 fits 32x32 map on 1024px screen perfectly
# Mouse wheel zooms between SCALE_MIN and SCALE_MAX
SCALE        = 0.5
SCALE_MIN    = 0.25
SCALE_MAX    = 1.0
SCALE_STEP   = 0.05

SCREEN_W    = 1024
SCREEN_H    = 720
TARGET_FPS  = 60

BG_COLOR         = (0, 0, 0)
SPRITE_COLOR_KEY = (244, 204, 161)

GRASS      = "GRASS"
REGOLITH   = "REGOLITH"
DIRT       = "DIRT"
ROCK       = "ROCK"

TILE_FILES: dict[str, str] = {
    DIRT:       "assets/tile_dirt.png",
    REGOLITH:   "assets/tile_regolith.png",
    GRASS:      "assets/tile_grass.png",
    ROCK:       "assets/tile_rock.png",
}

# ── Inspection Mode config ────────────────────────────────────────────────────

GHOST_OVERLAY_ALPHA  = 170
GHOST_SUBBLOCK_ALPHA = 140
SELECT_COLOR         = (255, 220, 50)
SELECT_WIDTH         = 2

# ── Billboard config ──────────────────────────────────────────────────────────

BILLBOARD_W      = 200
BILLBOARD_H      = 200
BILLBOARD_BG     = (15, 15, 15)
BILLBOARD_BORDER = (60, 60, 60)
BILLBOARD_TEXT   = (220, 220, 220)
BILLBOARD_ACCENT = (255, 220, 50)
LINE_COLOR       = (180, 180, 180)
BILLBOARD_OFFSET = (60, -110)

# ── World Generator ───────────────────────────────────────────────────────────

"""
 A lightweight world generator along side with the Manhattan disance model,
 to smooth out unrealistic constructions without the use of physics math.
 Block type propagation manager that uses defined layers to distinct between block types.
 These definitions can be found in the config header, make sure to verify the assets linkage before changing.
 
"""

def generate_heightmap(
    width: int = MAP_W, height: int = MAP_H, seed: int = SEED,
    num_peaks: int = 10, spread: float = 5.0,
) -> list[list[int]]:
    rng = random.Random(seed)
    peaks = [
        (rng.randint(0, width-1), rng.randint(0, height-1),
         rng.randint(MAX_HEIGHT // 2, MAX_HEIGHT))
        for _ in range(num_peaks)
    ]
    hmap: list[list[int]] = [[0] * height for _ in range(width)]
    for x in range(width):
        for y in range(height):
            best = 0
            for px, py, ph in peaks:
                dist = abs(x - px) + abs(y - py)
                value = max(0, ph - int(dist / spread))
                best = max(best, value)
            hmap[x][y] = min(MAX_HEIGHT, best)
    return hmap


def manhattan_decay(
    hmap: list[list[int]], max_slope: int = MAX_SLOPE,
) -> list[list[int]]:
    W, H = len(hmap), len(hmap[0])
    neighbours = [(-1,0),(1,0),(0,-1),(0,1)]
    changed = True
    while changed:
        changed = False
        for x in range(W):
            for y in range(H):
                for dx, dy in neighbours:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if hmap[x][y] > hmap[nx][ny] + max_slope:
                            hmap[x][y] = hmap[nx][ny] + max_slope
                            changed = True
    return hmap


def get_block_type(z: int, surface_z: int) -> str:
    if z == surface_z:
        if surface_z == HIGH_ALT:      return REGOLITH
        elif surface_z == MAX_HEIGHT:  return ROCK
        elif surface_z <= LOW_ALT:     return DIRT
        else:                           return GRASS
    elif z == surface_z - 1:
        return REGOLITH
    else:
        return ROCK if z <= 1 else DIRT


class World:
    def __init__(self, hmap: list[list[int]]):
        self.hmap   = hmap
        self.width  = len(hmap)
        self.height = len(hmap[0])

    def surface_z(self, x: int, y: int) -> int:
        return self.hmap[x][y]

    def column_blocks(self, x: int, y: int) -> list[tuple[int, str]]:
        sz = self.surface_z(x, y)
        return [(z, get_block_type(z, sz)) for z in range(sz + 1)]

    def block_stack_str(self, x: int, y: int) -> list[str]:
        return [bt for _, bt in reversed(self.column_blocks(x, y))]


def build_world(seed: int = SEED) -> World:
    hmap = generate_heightmap(seed=seed)
    hmap = manhattan_decay(hmap)
    return World(hmap)


# ── Sprite cache ──────────────────────────────────────────────────────────────

class SpriteCache:
    """
    Loads base sprites once and caches scaled versions by scale factor.
    On zoom change: scaled surfaces are regenerated from base, not re-scaled
    from already-scaled surfaces (avoids quality degradation).
    """

    def __init__(self):
        # Base surfaces at native 64x64 — never modified
        self._base: dict[str, pygame.Surface] = {}
        # Currently active scaled surfaces
        self._scaled: dict[str, pygame.Surface] = {}
        self._current_scale: float = 0.0

    def load(self) -> None:
        for block_type, path in TILE_FILES.items():
            surf = pygame.image.load(path).convert()
            surf.set_colorkey(SPRITE_COLOR_KEY)
            self._base[block_type] = surf

    def get(self, scale: float) -> dict[str, pygame.Surface]:
        """Returns scaled sprite dict, regenerating if scale changed."""
        if abs(scale - self._current_scale) > 0.001:
            self._rescale(scale)
        return self._scaled

    def _rescale(self, scale: float) -> None:
        tw = max(1, int(TILE_W_BASE * scale))
        th = max(1, int(TILE_H_BASE * scale))
        self._scaled = {}
        for block_type, base in self._base.items():
            scaled = pygame.transform.scale(base, (tw, th))
            scaled.set_colorkey(SPRITE_COLOR_KEY, pygame.RLEACCEL)
            self._scaled[block_type] = scaled
        self._current_scale = scale


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:
    """
    Isometric camera with WASD pan and mouse-wheel zoom.
    All projection math derived from current scale factor.
    """

    def __init__(self, screen_w: int, screen_h: int,
                 map_w: int = MAP_W, map_h: int = MAP_H,
                 move_speed: float = 4.0):
        self.screen_w   = screen_w
        self.screen_h   = screen_h
        self.map_w      = map_w
        self.map_h      = map_h
        self.move_speed = move_speed

        self.cam_x: float = 0.0
        self.cam_y: float = 0.0
        self.scale: float = SCALE

        self.moved = False        # pan moved
        self.zoomed = False       # scale changed

        self._icam_sx: int = 0
        self._icam_sy: int = 0

        self._derive_tile_dims()
        self._update_offsets()

    def _derive_tile_dims(self) -> None:
        """Recompute all tile dimension constants from current scale."""
        self.tile_w      = max(2, int(TILE_W_BASE * self.scale))
        self.tile_h      = max(2, int(TILE_H_BASE * self.scale))
        self.tile_z_step = max(1, int(TILE_Z_BASE * self.scale))
        self.tile_w_half    = self.tile_w // 2
        self.tile_h_quarter = self.tile_h // 4

    def _update_offsets(self) -> None:
        iso_w = (self.map_w + self.map_h) * self.tile_w_half
        iso_h = (self.map_w + self.map_h) * self.tile_h_quarter \
                + MAX_HEIGHT * self.tile_z_step + self.tile_h
        self.offset_x = (self.screen_w - iso_w) // 2 + self.map_h * self.tile_w_half
        self.offset_y = (self.screen_h - iso_h) // 2 \
                        + MAX_HEIGHT * self.tile_z_step + self.tile_h // 2

    def zoom(self, direction: int) -> None:
        """direction: +1 = zoom in, -1 = zoom out."""
        new_scale = round(self.scale + direction * SCALE_STEP, 4)
        new_scale = max(SCALE_MIN, min(SCALE_MAX, new_scale))
        if new_scale != self.scale:
            self.scale  = new_scale
            self.zoomed = True
            self._derive_tile_dims()
            self._update_offsets()

    def update(self, dt: float, keys) -> None:
        self.zoomed = False
        dx, dy = 0.0, 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:    dy -= 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  dy += 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  dx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: dx += 1.0

        if dx != 0.0 or dy != 0.0:
            if dx != 0.0 and dy != 0.0:
                dx *= 0.7071
                dy *= 0.7071
            self.cam_x += dx * self.move_speed * dt
            self.cam_y += dy * self.move_speed * dt
            self.cam_x = max(0.0, min(float(self.map_w), self.cam_x))
            self.cam_y = max(0.0, min(float(self.map_h), self.cam_y))
            self.moved = True
        else:
            self.moved = False

        self._icam_sx = int((self.cam_x - self.cam_y) * self.tile_w_half)
        self._icam_sy = int((self.cam_x + self.cam_y) * self.tile_h_quarter)

    def world_to_screen(self, wx: int, wy: int, wz: int) -> tuple[int, int]:
        sx = (wx - wy) * self.tile_w_half    - self._icam_sx + self.offset_x
        sy = (wx + wy) * self.tile_h_quarter - self._icam_sy \
             - wz * self.tile_z_step + self.offset_y
        return sx, sy

    def screen_to_world_float(self, sx: int, sy: int) -> tuple[float, float]:
        rx = sx - self.offset_x + self._icam_sx
        ry = sy - self.offset_y + self._icam_sy
        wx = (rx / self.tile_w_half    + ry / self.tile_h_quarter) * 0.5
        wy = (ry / self.tile_h_quarter - rx / self.tile_w_half)    * 0.5
        return wx, wy


# ── Mouse ──────────────────────────────────────────────────────────────

class Mouse:
    """
    Isometric mouse hit test with z-compensated reverse projection.
      For each candidate z level (0..MAX_HEIGHT), compensate the mouse
      screen_y by adding z*tile_z_step before reverse-projecting.
      This lands the search centre accurately on the correct world tile
      regardless of height. Search radius reduced to 2 — sufficient now
      that the centre estimate is accurate.

      The topmost hit (highest z whose top face contains the click) wins,
      which correctly matches painter's algorithm draw order.
    """
    _SEARCH_RADIUS = 2

    def __init__(self, world: World, camera: Camera):
        self.world  = world
        self.camera = camera

    def _point_in_diamond(self, mx: int, my: int,
                          wx: int, wy: int, wz: int) -> bool:
        sx, sy = self.camera.world_to_screen(wx, wy, wz)
        cx = sx + self.camera.tile_w_half
        cy = sy + self.camera.tile_h_quarter
        dx = mx - cx
        dy = my - cy
        hw = self.camera.tile_w_half
        qh = self.camera.tile_h_quarter
        if hw == 0 or qh == 0:
            return False
        proj1 = abs( dx / hw + dy / qh)
        proj2 = abs(-dx / hw + dy / qh)
        return proj1 <= 1.0 and proj2 <= 1.0

    def _reverse_project_at_z(self, mx: int, my: int,
                               wz: int) -> tuple[float, float]:
        """
        Reverse-project screen (mx, my) assuming the clicked tile is at
        height wz. Compensates sy by +wz*tile_z_step before inverting,
        which cancels the upward screen shift caused by that height.
        """
        cam = self.camera
        compensated_sy = my + wz * cam.tile_z_step
        return cam.screen_to_world_float(mx, compensated_sy)

    def pick(self, mx: int, my: int) -> tuple[int, int, int] | None:
        r      = self._SEARCH_RADIUS
        best: tuple[int, int, int] | None = None
        best_z = -1

        # Try each possible z level — compensate reverse projection for each
        for candidate_z in range(MAX_HEIGHT + 1):
            fx, fy = self._reverse_project_at_z(mx, my, candidate_z)
            cx, cy = int(fx), int(fy)

            for tx in range(cx - r, cx + r + 1):
                for ty in range(cy - r, cy + r + 1):
                    if tx < 0 or ty < 0 or tx >= self.world.width or ty >= self.world.height:
                        continue
                    tz = self.world.surface_z(tx, ty)
                    # Only test tiles whose actual surface matches this z level
                    if tz != candidate_z:
                        continue
                    if tz <= best_z:
                        continue
                    if self._point_in_diamond(mx, my, tx, ty, tz):
                        best_z = tz
                        best   = (tx, ty, tz)

        return best


# ── Lightweight UI Components ─────────────────────────────────────────────────────────────────

class Billboard:
    def __init__(self):
        self._font_large = None
        self._font_small = None
        self.selected: tuple[int, int, int] | None = None
        self._cell_data: dict = {}

    def init_fonts(self) -> None:
        try:
            self._font_large = pygame.font.SysFont("couriernew", 13, bold=True)
            self._font_small = pygame.font.SysFont("couriernew", 11)
        except Exception:
            self._font_large = pygame.font.SysFont("monospace", 13, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 11)

    def set_selection(self, tile: tuple[int, int, int] | None,
                      world: World) -> None:
        self.selected = tile
        if tile is not None:
            wx, wy, wz = tile
            stack = world.block_stack_str(wx, wy)
            self._cell_data = {
                "coord":  f"[{wx}, {wy}]",
                "height": str(wz),
                "soil":   stack[0] if stack else "?",
                "stack":  stack[:4],
            }

    def _anchor_point(self, camera: Camera,
                      wx: int, wy: int, wz: int) -> tuple[int, int]:
        sx, sy = camera.world_to_screen(wx, wy, wz)
        return sx + camera.tile_w, sy + camera.tile_h_quarter

    def _panel_pos(self, anchor: tuple[int, int]) -> tuple[int, int]:
        ax, ay = anchor
        ox, oy = BILLBOARD_OFFSET
        px, py = ax + ox, ay + oy
        if px + BILLBOARD_W > SCREEN_W - 10:
            px = ax - BILLBOARD_W - abs(ox)
        if py < 10:
            py = ay + abs(oy) // 2
        if py + BILLBOARD_H > SCREEN_H - 10:
            py = SCREEN_H - BILLBOARD_H - 10
        return px, py

    def draw(self, screen: pygame.Surface, camera: Camera) -> None:
        if self.selected is None or self._font_small is None:
            return
        wx, wy, wz = self.selected
        anchor     = self._anchor_point(camera, wx, wy, wz)
        px, py     = self._panel_pos(anchor)

        panel_mid_y  = py + BILLBOARD_H // 2
        panel_edge_x = px if anchor[0] > px + BILLBOARD_W // 2 else px + BILLBOARD_W
        pygame.draw.line(screen, LINE_COLOR, anchor, (panel_edge_x, panel_mid_y), 1)
        pygame.draw.circle(screen, SELECT_COLOR, anchor, 3)

        shadow = pygame.Surface((BILLBOARD_W, BILLBOARD_H), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 120))
        screen.blit(shadow, (px + 3, py + 3))

        panel_rect = pygame.Rect(px, py, BILLBOARD_W, BILLBOARD_H)
        pygame.draw.rect(screen, BILLBOARD_BG, panel_rect)
        pygame.draw.rect(screen, BILLBOARD_BORDER, panel_rect, 1)
        pygame.draw.rect(screen, (30,30,30), pygame.Rect(px, py, BILLBOARD_W, 18))
        pygame.draw.line(screen, BILLBOARD_ACCENT,
                         (px, py+18), (px+BILLBOARD_W, py+18), 1)

        pad, lh = 8, 16
        tx, ty  = px + pad, py + 4

        screen.blit(
            self._font_large.render(
                f"CELL  {self._cell_data['coord']}", True, BILLBOARD_ACCENT),
            (tx, ty))
        ty += 20
        pygame.draw.line(screen, BILLBOARD_BORDER,
                         (px+pad, ty), (px+BILLBOARD_W-pad, ty), 1)
        ty += 6

        def row(label: str, value: str, highlight: bool = False) -> None:
            nonlocal ty
            col = BILLBOARD_ACCENT if highlight else (120,120,120)
            screen.blit(self._font_small.render(label, True, col),           (tx, ty))
            screen.blit(self._font_small.render(value, True, BILLBOARD_TEXT),(tx+70, ty))
            ty += lh

        row("height :", self._cell_data["height"])
        row("soil   :", self._cell_data["soil"], highlight=True)
        ty += 2
        pygame.draw.line(screen, BILLBOARD_BORDER,
                         (px+pad, ty), (px+BILLBOARD_W-pad, ty), 1)
        ty += 5
        screen.blit(self._font_small.render("stack  :", True, (120,120,120)), (tx, ty))
        ty += lh
        for i, bt in enumerate(self._cell_data["stack"]):
            prefix = ">> " if i == 0 else "   "
            col    = BILLBOARD_TEXT if i == 0 else (80,80,80)
            screen.blit(self._font_small.render(f"  {prefix}{bt}", True, col), (tx, ty))
            ty += lh


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    Scale-aware dirty-flag renderer with Inspection Mode.
    On zoom: sprites rescaled from base, screen positions recomputed,
             column_sprites rebuilt — full redraw triggered automatically.
    """

    def __init__(self, world: World, camera: Camera,
                 sprite_cache: SpriteCache):
        self.world        = world
        self.camera       = camera
        self.sprite_cache = sprite_cache

        self.surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self.surface.fill(BG_COLOR)

        self._ghost_overlay  = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self._ghost_overlay.fill((0, 0, 0, GHOST_OVERLAY_ALPHA))
        self._ghost_col_surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)

        self._selected: tuple[int, int, int] | None = None
        self.inspection_mode: bool = False

        # Precomputed draw order — never changes
        self._draw_order: list[tuple[int, int]] = sorted(
            [(x, y) for x in range(world.width) for y in range(world.height)],
            key=lambda t: t[0] + t[1]
        )

        # Scale-dependent — rebuilt on zoom
        self._column_sprites: list[list[list[tuple[int, pygame.Surface]]]] = []
        self._sx:      list[list[int]] = [[0]*world.height for _ in range(world.width)]
        self._base_sy: list[list[int]] = [[0]*world.height for _ in range(world.width)]

        self._dirty: set[tuple[int, int]] = set(self._draw_order)
        self._rebuild_column_sprites()
        self._recompute_screen_positions()

    def _rebuild_column_sprites(self) -> None:
        """Rebuild (z, Surface) pairs using current scaled sprites."""
        sprites = self.sprite_cache.get(self.camera.scale)
        self._column_sprites = [
            [
                [(z, sprites[bt]) for z, bt in self.world.column_blocks(x, y)]
                for y in range(self.world.height)
            ]
            for x in range(self.world.width)
        ]

    def _recompute_screen_positions(self) -> None:
        wts = self.camera.world_to_screen
        for x in range(self.world.width):
            for y in range(self.world.height):
                sx, sy          = wts(x, y, 0)
                self._sx[x][y]      = sx
                self._base_sy[x][y] = sy

    def set_selection(self, tile: tuple[int, int, int] | None) -> None:
        if self._selected is not None:
            self._dirty.add((self._selected[0], self._selected[1]))
        self._selected       = tile
        self.inspection_mode = tile is not None
        if tile is not None:
            self._dirty.add((tile[0], tile[1]))
        self._mark_all_dirty()

    def mark_dirty(self, x: int, y: int) -> None:
        self._dirty.add((x, y))

    def _mark_all_dirty(self) -> None:
        self._dirty = set(self._draw_order)

    def _draw_selection_diamond(self, target: pygame.Surface,
                                wx: int, wy: int, wz: int) -> None:
        sx  = self._sx[wx][wy]
        sy  = self._base_sy[wx][wy] - wz * self.camera.tile_z_step
        hw  = self.camera.tile_w_half
        qh  = self.camera.tile_h_quarter
        tw  = self.camera.tile_w
        pygame.draw.lines(target, SELECT_COLOR, True, [
            (sx + hw, sy),
            (sx + tw, sy + qh),
            (sx + hw, sy + qh * 2),
            (sx,      sy + qh),
        ], SELECT_WIDTH)

    def _blit_selected_column_full(self, target: pygame.Surface) -> None:
        if self._selected is None:
            return
        wx, wy, wz = self._selected
        sx         = self._sx[wx][wy]
        base_sy    = self._base_sy[wx][wy]
        z_step     = self.camera.tile_z_step
        col        = self._column_sprites[wx][wy]

        self._ghost_col_surf.fill((0, 0, 0, 0))
        for z, sprite in col:
            sy = base_sy - z * z_step
            if z == wz:
                target.blit(sprite, (sx, sy))
            else:
                self._ghost_col_surf.blit(sprite, (sx, sy))

        self._ghost_col_surf.set_alpha(GHOST_SUBBLOCK_ALPHA)
        target.blit(self._ghost_col_surf, (0, 0))

    def update(self, screen: pygame.Surface) -> None:
        # Zoom change, rebuild sprites and positions, full redraw
        if self.camera.zoomed:
            self._rebuild_column_sprites()
            self._recompute_screen_positions()
            self._mark_all_dirty()

        # Pan, recompute positions, full redraw
        if self.camera.moved:
            self._recompute_screen_positions()
            self._mark_all_dirty()

        if self._dirty:
            self.surface.fill(BG_COLOR)
            dirty       = self._dirty
            draw_order  = self._draw_order
            col_sprites = self._column_sprites
            sx_table    = self._sx
            sy_table    = self._base_sy
            blit        = self.surface.blit
            sw, sh      = SCREEN_W, SCREEN_H
            tw          = self.camera.tile_w
            th          = self.camera.tile_h
            z_step      = self.camera.tile_z_step
            cull_x      = tw
            cull_y      = th + MAX_HEIGHT * z_step

            for (x, y) in draw_order:
                if (x, y) not in dirty:
                    continue
                sx      = sx_table[x][y]
                base_sy = sy_table[x][y]
                if sx < -cull_x or sx > sw + cull_x:
                    continue
                if base_sy < -cull_y or base_sy > sh + cull_y:
                    continue
                for z, sprite in col_sprites[x][y]:
                    blit(sprite, (sx, base_sy - z * z_step))

            self._dirty.clear()

        # Composite to screen
        if self.inspection_mode and self._selected is not None:
            screen.blit(self.surface, (0, 0))
            screen.blit(self._ghost_overlay, (0, 0))
            self._blit_selected_column_full(screen)
            self._draw_selection_diamond(screen,
                                         self._selected[0],
                                         self._selected[1],
                                         self._selected[2])
        else:
            screen.blit(self.surface, (0, 0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Voxel Engine  —  seed {SEED}")
    clock = pygame.time.Clock()

    world        = build_world(seed=SEED)
    sprite_cache = SpriteCache()
    sprite_cache.load()

    camera    = Camera(SCREEN_W, SCREEN_H, move_speed=4.0)
    renderer  = Renderer(world, camera, sprite_cache)
    picker    = Mouse(world, camera)
    billboard = Billboard()
    billboard.init_fonts()

    print(f"World {world.width}x{world.height} | seed={SEED}")
    
    running = True
    while running:
        dt   = clock.tick(TARGET_FPS) / 1000.0
        keys = pygame.key.get_pressed()
        camera.update(dt, keys)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(event.y)        # +1 scroll up = zoom in
                # Keep selection valid after zoom (positions shifted)
                if renderer._selected is not None:
                    renderer.set_selection(renderer._selected)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if renderer.inspection_mode:
                        renderer.set_selection(None)
                        billboard.set_selection(None, world)
                    else:
                        running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    hit = picker.pick(mx, my)
                    if hit is not None:
                        renderer.set_selection(hit)
                        billboard.set_selection(hit, world)

        if camera.moved:
            renderer.set_selection(renderer._selected)

        screen.fill(BG_COLOR)
        renderer.update(screen)
        billboard.draw(screen, camera)

        fps = clock.get_fps()
        mode_str = "  [INSPECTION]" if renderer.inspection_mode else ""
        pygame.display.set_caption(
            f"Voxel Engine [seed {SEED} | scale {camera.scale:.2f} | {fps:.0f} fps]{mode_str}"
        )
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()