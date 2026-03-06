from __future__ import annotations
import random
import math
import pygame
import sys
import time

from enum import Enum, auto
from dataclasses import dataclass
"""
Project: VX World

A voxel world build for lightweight GA agents to test on.
the world map is designed to be traversable and also adhere
to some primitive environment factors, like time, season and thermo dynamics (decay gradients)

Features:
- Voxel Engine (done)
- Map Generator (done)
- Simulation Manager (simple timeline epoch for seasons and day/night cycles) * unfinished state *
- Decay System (todo)

"""

# ── Config ────────────────────────────────────────────────────────────────────

SEED        = random.randint(2,2048)
MAP_W       = 24
MAP_H       = 24
MAX_HEIGHT  = 8
HIGH_ALT    = 5
LOW_ALT     = 1
MAX_SLOPE   = 1

WORLD_PEAKS = 1
WORLD_SPREAD = 8.0

# Base sprite dimensions (should never change)
TILE_W_BASE = 128
TILE_H_BASE = 128
TILE_Z_BASE = 64

# Scale factors
SCALE        = 0.5
SCALE_MIN    = 0.25
SCALE_MAX    = 1.0
SCALE_STEP   = 0.05

SCREEN_W    = 1024
SCREEN_H    = 720
TARGET_FPS  = 60

BG_COLOR = (0, 0, 0)

GRASS      = "GRASS"
REGOLITH   = "REGOLITH"
DIRT       = "DIRT"
ROCK       = "ROCK"

# ── Texture Engine config ─────────────────────────────────────────────────────

# Native resolution the collapse runs at — scaled up with nearest-neighbor on blit.
# smaller = chunkier pixels, faster collapse. 12 is a good GB-feel sweet spot.
TEX_NATIVE_W = 12
TEX_NATIVE_H = 8

# 4-color palette — index 0=darkest shadow, 3=lightest catch-light
PALETTE: list[tuple[int,int,int]] = [
    (15,  40,  15),   # 0  darkest  — deep shadow, rock base
    (48,  80,  48),   # 1  dark     — mid tone, subsurface
    (100, 140,  60),  # 2  light    — surface colour, grass body
    (160, 200,  80),  # 3  lightest — catch-light, top highlight
]

# Face shading: top=full texture,
# left/right=flat palette index
FACE_LEFT_IDX  = 1   # dark face
FACE_RIGHT_IDX = 0   # darkest face

# ── Inspection Mode config ────────────────────────────────────────────────────

GHOST_OVERLAY_ALPHA  = 70
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

# ── Sim config ──────────────────────────────────────────────────────────

YEAR_LENGTH    = 120     # epochs per full year  (120s = 2 min real-time)
AVG_TEMP       = 12.0    # °C — mid-year baseline temperature
TEMP_AMP       = 10.0    # °C — swing above/below baseline

EPOCHS_PER_DAY          = 4    # How many epochs make one full day cycle
REAL_SECONDS_PER_DAY    = 30.0 # 30s, 60s, 120s, ~
EPOCH_DURATION          = REAL_SECONDS_PER_DAY / EPOCHS_PER_DAY

# ── World Generator ───────────────────────────────────────────────────────────

"""
 A lightweight world generator along side with the Manhattan disance model,
 to smooth out unrealistic constructions without the use of physics math.
 Block type propagation manager that uses defined layers to distinct between block types.
 These definitions can be found in the config header, make sure to verify the assets linkage before changing.
 
"""

def generate_heightmap(
    width: int = MAP_W, height: int = MAP_H, seed: int = SEED,
    num_peaks: int = WORLD_PEAKS, spread: float = WORLD_SPREAD,
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
    print(f'Generating new world...[{SEED}]')
    hmap = generate_heightmap(seed=seed)
    hmap = manhattan_decay(hmap)
    return World(hmap)


# ── Texture Engine ────────────────────────────────────────────────────────────

@dataclass
class MaterialDef:
    """
    Defines the visual personality of a block type.
    All values tunable independently — no other code needs to change.

    bias        : probability weight per palette index [dark→light].
                  Does not need to sum to 1.0 — normalised internally.
    coherence   : 0.0–1.0. How strongly a collapsed cell pulls its
                  neighbours toward the same colour. High = smooth
                  blobby patches. Low = noisy, craggy.
    run_pref    : 0.0–1.0. Tendency to continue the same colour along
                  a scan row. Produces grain/striations at high values.
    seed_offset : mixed into the world SEED so each material always
                  produces a distinct pattern even at the same world seed.
    """
    bias:        list[float]
    coherence:   float
    run_pref:    float
    seed_offset: int = 0


MATERIALS: dict[str, MaterialDef] = {

    # Soft organic patches — living surface texture.
    # High coherence pools colour into blobs rather than scatter.
    # Low run_pref keeps it isotropic (no grain direction).
    # Bias pushes into p2 (light green) with p3 catch-light highlights.
    GRASS: MaterialDef(
        bias        = [0.03, 0.12, 0.55, 0.30],
        coherence   = 0.75,
        run_pref    = 0.15,
        seed_offset = 7,
    ),

    # Compressed, muted, earthy — p3 (highlight) nearly absent.
    # Mid coherence keeps it quiet. Moderate run_pref gives subtle
    # horizontal grain — reads as sediment / packed soil.
    DIRT: MaterialDef(
        bias        = [0.08, 0.62, 0.28, 0.02],
        coherence   = 0.55,
        run_pref    = 0.35,
        seed_offset = 13,
    ),

    # Dark, fractured, high contrast — bimodal between p0 and p1.
    # Very low coherence = craggy, cells rarely clump.
    # High run_pref = fracture lines run horizontally across the face,
    # like real rock cleavage. Reads unmistakably as stone.
    ROCK: MaterialDef(
        bias        = [0.50, 0.30, 0.15, 0.05],
        coherence   = 0.20,
        run_pref    = 0.65,
        seed_offset = 23,
    ),

    # Banded strata — geological layering feel.
    # Bias stays in p1–p2 (mid-tones only), no extremes.
    # Very high run_pref drives strong horizontal bands that read as
    # compressed dust / ash / fine sediment layers.
    REGOLITH: MaterialDef(
        bias        = [0.10, 0.50, 0.32, 0.08],
        coherence   = 0.40,
        run_pref    = 0.72,
        seed_offset = 37,
    ),

}

class TextureEngine:
    """
    Procedural tile generator using a single-pass collapse function.

    Two-tier cache — designed around the dirty-flag model:

      Tier 1  _tex_cache  : block_type → raw pixel grid (palette indices)
                            Built once at load(). Invalidated only on world
                            state change (decay, mutation) via invalidate().

      Tier 2  _surf_cache : (block_type, scale_key) → pygame.Surface
                            Pre-baked for ALL discrete zoom levels at load()
                            so runtime zoom is a pure dict lookup — zero cost.

    Public interface identical to old SpriteCache so Renderer is untouched.
    """

    def __init__(self, seed: int = SEED) -> None:
        self._seed       = seed
        self._tex_cache:  dict[str, list[list[int]]]            = {}
        self._surf_cache: dict[tuple[str, int], pygame.Surface] = {}
        self._current_scale: float = 0.0
        self._scaled_dict:   dict[str, pygame.Surface]          = {}

        # Pre-compute every discrete zoom level that the camera can reach.
        # SCALE_STEP increments from SCALE_MIN to SCALE_MAX → 16 levels.
        # Stored as integer keys (scale * 1000) to avoid float drift.
        self._zoom_levels: list[int] = []
        s = SCALE_MIN
        while s <= SCALE_MAX + 0.001:
            self._zoom_levels.append(round(s * 1000))
            s = round(s + SCALE_STEP, 4)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Full startup bake — runs once.
        Phase 1: collapse all materials into pixel grids  (trivial, ~2ms)
        Phase 2: build pygame Surfaces at every zoom level (~60ms typical)
        After this returns, all runtime paths are pure cache lookups.
        """
        n_mat   = len(MATERIALS)
        n_zoom  = len(self._zoom_levels)
        total   = n_mat + n_mat * n_zoom
        
        print(f'Generating textures...')
        print(f'    Materials   = {n_mat} [depth={n_zoom}]')
        print(f'    Surfaces    = {n_mat * n_zoom}')
        
        t_start = time.perf_counter()

        # ── Phase 1: collapse ─────────────────────────────────────────────
        for bt in MATERIALS:
            self._tex_cache[bt] = self._collapse(bt)

        # ── Phase 2: bake all zoom levels ─────────────────────────────────
        for scale_key in self._zoom_levels:
            scale = scale_key / 1000.0
            tw    = max(2, int(TILE_W_BASE * scale))
            th    = max(2, int(TILE_H_BASE * scale))
            # Build shared diamond mask for this size (reused across materials)
            mask  = self._make_diamond_mask(tw, th)
            for bt in MATERIALS:
                surf = self._build_tile_surface(
                    self._tex_cache[bt], tw, th, mask
                )
                self._surf_cache[(bt, scale_key)] = surf

        elapsed = (time.perf_counter() - t_start) * 1000
        print(f'  baked {len(self._surf_cache)} surfaces in {elapsed:.0f}ms\n')

        # Prime the active dict for the default scale
        self._activate_scale(SCALE)

    def get(self, scale: float) -> dict[str, pygame.Surface]:
        """Runtime zoom — O(1) dict lookup, no surface construction."""
        if abs(scale - self._current_scale) > 0.001:
            self._activate_scale(scale)
        return self._scaled_dict

    def invalidate(self, block_type: str) -> None:
        """
        Force re-collapse + re-bake for one material.
        Called by the decay system when a block type mutates.
        Rebuilds all zoom levels for that material, then re-activates
        the current scale so the next frame picks up the change.
        """
        print(f'TextureEngine.invalidate({block_type})')
        self._tex_cache[block_type] = self._collapse(block_type)

        for scale_key in self._zoom_levels:
            scale = scale_key / 1000.0
            tw    = max(2, int(TILE_W_BASE * scale))
            th    = max(2, int(TILE_H_BASE * scale))
            mask  = self._make_diamond_mask(tw, th)
            surf  = self._build_tile_surface(
                self._tex_cache[block_type], tw, th, mask
            )
            self._surf_cache[(block_type, scale_key)] = surf

        self._activate_scale(self._current_scale)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _activate_scale(self, scale: float) -> None:
        """Swap the active surface dict to the requested zoom level."""
        scale_key = round(scale * 1000)
        # Snap to nearest known key (handles floating-point edge cases)
        if scale_key not in [k for _, k in self._surf_cache]:
            scale_key = min(self._zoom_levels, key=lambda k: abs(k - scale_key))
        self._scaled_dict   = {
            bt: self._surf_cache[(bt, scale_key)]
            for bt in MATERIALS
            if (bt, scale_key) in self._surf_cache
        }
        self._current_scale = scale

    # ── Collapse function ─────────────────────────────────────────────────────

    def _collapse(self, block_type: str) -> list[list[int]]:
        """
        Single-pass collapse over a TEX_NATIVE_W × TEX_NATIVE_H grid.

        Sweeps left-to-right, top-to-bottom. At each cell:
          1. Start from material bias distribution.
          2. Left + top neighbours propagate constraint (scaled by coherence).
          3. Run preference boosts continuation of same colour along a row.
          4. Sample from the final weighted distribution.

        One pass only — no iteration. Character comes entirely from MaterialDef.
        """
        mat  = MATERIALS[block_type]
        rng  = random.Random(self._seed ^ mat.seed_offset)
        W, H = TEX_NATIVE_W, TEX_NATIVE_H
        n    = len(PALETTE)

        total        = sum(mat.bias)
        base_weights = [b / total for b in mat.bias]
        grid: list[list[int]] = [[0] * W for _ in range(H)]

        for row in range(H):
            prev_col = -1
            for col in range(W):
                weights = list(base_weights)

                if col > 0:
                    li = grid[row][col - 1]
                    for i in range(n):
                        pull = mat.coherence if i == li else -mat.coherence / (n - 1)
                        weights[i] = max(0.0, weights[i] + pull * weights[i])

                if row > 0:
                    ti = grid[row - 1][col]
                    for i in range(n):
                        pull = mat.coherence if i == ti else -mat.coherence / (n - 1)
                        weights[i] = max(0.0, weights[i] + pull * weights[i])

                if prev_col >= 0:
                    weights[prev_col] = max(
                        0.0, weights[prev_col] + mat.run_pref * weights[prev_col]
                    )

                wsum    = sum(weights) or 1.0
                weights = [w / wsum for w in weights]
                r       = rng.random()
                cum     = 0.0
                chosen  = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break

                grid[row][col] = chosen
                prev_col = chosen

        return grid

    # ── Surface builders ──────────────────────────────────────────────────────

    @staticmethod
    def _make_diamond_mask(tw: int, th: int) -> pygame.Surface:
        """
        White diamond on transparent background — shared across all materials
        at the same tile size to avoid redundant surface allocation.
        """
        hw   = tw // 2
        qh   = max(1, th // 4)
        mask = pygame.Surface((tw, th), pygame.SRCALPHA)
        mask.fill((0, 0, 0, 0))
        pygame.draw.polygon(mask, (255, 255, 255, 255), [
            (hw, 0), (tw - 1, qh), (hw, qh * 2), (0, qh)
        ])
        return mask

    def _build_tile_surface(
        self,
        grid: list[list[int]],
        tw: int, th: int,
        mask: pygame.Surface,
    ) -> pygame.Surface:
        """
        Composite one isometric tile surface (tw × th, SRCALPHA).

        Faces:
          Top   — collapse texture clipped to diamond via BLEND_RGBA_MULT
          Left  — flat FACE_LEFT_IDX  colour (left parallelogram)
          Right — flat FACE_RIGHT_IDX colour (right parallelogram)
        """
        hw = tw // 2
        qh = max(1, th // 4)
        tz = max(1, int(TILE_Z_BASE * (tw / TILE_W_BASE)))

        # ── Top face ──────────────────────────────────────────────────────
        native = pygame.Surface((TEX_NATIVE_W, TEX_NATIVE_H))
        pxa    = pygame.PixelArray(native)
        for row in range(TEX_NATIVE_H):
            for col in range(TEX_NATIVE_W):
                pxa[col, row] = native.map_rgb(*PALETTE[grid[row][col]])
        del pxa

        tex  = pygame.transform.scale(native, (tw, qh * 2))
        temp = pygame.Surface((tw, th), pygame.SRCALPHA)
        temp.blit(tex, (0, 0))
        temp.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        surf = pygame.Surface((tw, th), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        surf.blit(temp, (0, 0))

        # ── Left face ─────────────────────────────────────────────────────
        pygame.draw.polygon(surf, (*PALETTE[FACE_LEFT_IDX], 255), [
            (0,  qh),       (hw, qh * 2),
            (hw, qh*2 + tz),(0,  qh  + tz),
        ])

        # ── Right face ────────────────────────────────────────────────────
        pygame.draw.polygon(surf, (*PALETTE[FACE_RIGHT_IDX], 255), [
            (hw,     qh * 2),   (tw - 1, qh),
            (tw - 1, qh  + tz), (hw,     qh*2 + tz),
        ])

        return surf


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


# ── Simulation Manager ─────────────────────────────────────────────────────

class Season(Enum):
    SPRING = auto()
    SUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()


@dataclass
class TimelineState:
    """Immutable snapshot of timeline state — safe to hand to any system."""
    epoch:       int
    year:        int
    year_epoch:  int        # position within current year (0..year_length-1)
    season:      Season
    base_temp:   float      # °C derived from sinusoidal curve
    paused:      bool


class Timeline:
    """
    Decoupled simulation clock. Knows nothing about the world,
    renderer, or any other system.

    Ticks in real-time: accumulates dt seconds, fires one epoch
    per EPOCH_DURATION seconds regardless of FPS.

    Year cycle:
        year_length epochs split into 4 equal seasons.
        Temperature follows a sinusoidal curve — cold at epoch 0
        (start of spring, rising), peak at summer, trough at winter.

    Controls:
        spacebar — pause / resume
    """

    _SEASON_MAP: dict[int, Season] = {}   # built in __init__

    def __init__(
        self,
        year_length:    int   = YEAR_LENGTH,
        avg_temp:       float = AVG_TEMP,
        temp_amp:       float = TEMP_AMP,
        epoch_duration: float = EPOCH_DURATION,
    ) -> None:
        self.year_length    = year_length
        self.avg_temp       = avg_temp
        self.temp_amp       = temp_amp
        self.epoch_duration = epoch_duration

        self._epoch:       int   = 0
        self._accumulator: float = 0.0
        self.paused:       bool  = False

        # Pre-build season lookup for O(1) access
        sl = year_length // 4
        self._season_boundaries = [
            (sl * 0, sl * 1, Season.SPRING),
            (sl * 1, sl * 2, Season.SUMMER),
            (sl * 2, sl * 3, Season.AUTUMN),
            (sl * 3, year_length, Season.WINTER),
        ]

    # Helpers
    
    def phase_to_multipliers(self, phase: float) -> tuple[float, float, float]:
        """Return (r_mult, g_mult, b_mult) for given day_phase."""
        # Predefined anchors: midnight, dawn, noon, dusk, midnight
        anchors = [
            (0.00, (0.20, 0.20, 0.40)),  # night: cool/dim
            (0.25, (1.00, 0.70, 0.40)),  # dawn: warm
            (0.50, (1.00, 1.00, 1.00)),  # noon: neutral
            (0.75, (1.00, 0.50, 0.60)),  # dusk: reddish
            (1.00, (0.20, 0.20, 0.40)),  # night again
        ]
        # Find surrounding anchors and lerp
        for i in range(len(anchors)-1):
            p0, c0 = anchors[i]
            p1, c1 = anchors[i+1]
            if p0 <= phase <= p1:
                t = (phase - p0) / (p1 - p0) if p1 != p0 else 0
                return tuple(c0[j] + t * (c1[j] - c0[j]) for j in range(3))
        return anchors[-1][1]

    # API'S

    @property
    def day_phase(self) -> float:
        """
        Smooth 0.0-1.0 day cycle.
        0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
        """
        day_epoch = self._epoch % EPOCHS_PER_DAY
        # Interpolate progress within current epoch (0.0 → 1.0)
        progress = min(1.0, self._accumulator / self.epoch_duration) if self.epoch_duration > 0 else 0.0
        return (day_epoch + progress) / EPOCHS_PER_DAY
    
    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def year(self) -> int:
        return self._epoch // self.year_length

    @property
    def year_epoch(self) -> int:
        return self._epoch % self.year_length

    @property
    def season(self) -> Season:
        ye = self.year_epoch
        for start, end, s in self._season_boundaries:
            if start <= ye < end:
                return s
        return Season.WINTER

    @property
    def base_temp(self) -> float:
        """
        Sinusoidal temperature curve over the year.
        Starts at avg (spring rising), peaks at summer, troughs at winter.
        Phase offset -π/2 so epoch 0 = spring equinox (temp rising).
        """
        angle = 2 * math.pi * (self.year_epoch / self.year_length) - math.pi / 2
        return round(self.avg_temp + self.temp_amp * math.sin(angle), 1)

    def state(self) -> TimelineState:
        """Return an immutable snapshot for other systems to read."""
        return TimelineState(
            epoch      = self._epoch,
            year       = self.year,
            year_epoch = self.year_epoch,
            season     = self.season,
            base_temp  = self.base_temp,
            paused     = self.paused,
        )

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def tick(self, dt: float) -> bool:
        """
        Advance accumulator by dt seconds.
        Returns True if one or more epochs fired this frame.
        Epochs are skipped silently if paused.
        """
        if self.paused:
            return False
        self._accumulator += dt
        fired = False
        while self._accumulator >= self.epoch_duration:
            self._accumulator -= self.epoch_duration
            self._epoch += 1
            fired = True
        return fired

# ── GUI ─────────────────────────────────────────────────────

class Information:
    """
    reads from Timeline.state()

    Layout:
    ┌────────────────────────────────────────────────────────────────┐
    │  EPOCH  000,001+ | phase │  [season]  │  0.0°C  │  [extras]    │
    └────────────────────────────────────────────────────────────────┘
    """

    # Season display config: label, colour tint
    _SEASON_STYLE: dict[Season, tuple[str, tuple[int,int,int]]] = {
        Season.SPRING: ("SPRING", (120, 200, 100)),
        Season.SUMMER: ("SUMMER", (240, 200,  60)),
        Season.AUTUMN: ("AUTUMN", (210, 120,  40)),
        Season.WINTER: ("WINTER", ( 90, 160, 220)),
    }

    _PANEL_W  = 460
    _PANEL_H  = 28
    _PANEL_Y  = 8          # pixels from top of screen
    _BG       = (12, 12, 12)
    _BORDER   = (55, 55, 55)
    _TEXT     = (200, 200, 200)
    _ACCENT   = (255, 220, 50)
    _DIVIDER  = (50, 50, 50)

    def __init__(self, timeline: Timeline, screen_w: int = SCREEN_W) -> None:
        self.timeline  = timeline
        self.screen_w  = screen_w
        self._font     = None
        self._surf     = pygame.Surface((self._PANEL_W, self._PANEL_H),
                                        pygame.SRCALPHA)

    def init_fonts(self) -> None:
        try:
            self._font = pygame.font.SysFont("couriernew", 12, bold=True)
        except Exception:
            self._font = pygame.font.SysFont("monospace", 12, bold=True)

    def _divider(self, x: int) -> None:
        pygame.draw.line(self._surf, self._DIVIDER,
                         (x, 4), (x, self._PANEL_H - 4), 1)

    def draw(self, screen: pygame.Surface) -> None:
        if self._font is None:
            return

        state = self.timeline.state()
        s     = self._surf
        s.fill((0, 0, 0, 0))

        # Panel background + border
        panel_rect = pygame.Rect(0, 0, self._PANEL_W, self._PANEL_H)
        pygame.draw.rect(s, (*self._BG, 210), panel_rect, border_radius=3)
        pygame.draw.rect(s, (*self._BORDER, 255), panel_rect, 1, border_radius=3)

        pad = 10
        cy  = self._PANEL_H // 2   # vertical centre

        def text(txt: str, x: int, color: tuple) -> int:
            """Blit text at (x, cy-aligned), return right edge x."""
            surf = self._font.render(txt, True, color)
            s.blit(surf, (x, cy - surf.get_height() // 2))
            return x + surf.get_width()

        x = pad

        # Epoch
        epoch_str = f"EP {state.epoch:>07,} DP {self.timeline.day_phase:.3f}"
        
        x = text(epoch_str, x, self._ACCENT)
        x += 10
        self._divider(x)
        x += 10

        # Season
        s_label, s_color = self._SEASON_STYLE[state.season]
        x = text(s_label, x, s_color)
        x += 10
        self._divider(x)
        x += 10

        # Temperature
        temp_str = f"{state.base_temp:+.1f}\u00b0C"
        
        # Cool → warm colour interpolation
        t = (state.base_temp - (AVG_TEMP - TEMP_AMP)) / (2 * TEMP_AMP)
        t = max(0.0, min(1.0, t))
        temp_color = (
            int(60  + t * 195),   # R: 60→255
            int(160 - t * 100),   # G: 160→60
            int(220 - t * 180),   # B: 220→40
        )
        x = text(temp_str, x, temp_color)
        x += 10
        self._divider(x)
        x += 10

        # Pause / play indicator
        play_sym = "PAUSED" if state.paused else "LIVE"
        play_col = (180, 80, 80) if state.paused else (80, 200, 80)
        text(play_sym, x, play_col)

        # Blit panel centered at top of screen
        panel_x = (self.screen_w - self._PANEL_W) // 2
        screen.blit(s, (panel_x, self._PANEL_Y))


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    Scale-aware dirty-flag renderer with Inspection Mode.
    On zoom: sprites rescaled from base, screen positions recomputed,
             column_sprites rebuilt — full redraw triggered automatically.
    """

    def __init__(self, world: World, camera: Camera,
                 sprite_cache: TextureEngine):
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
        # old approach: key=lambda t: t[0] + t[1]
        # new approach: primary -> depth, secondary -> y breaks ties
        self._draw_order: list[tuple[int, int]] = sorted(
            [(x, y) for x in range(world.width) for y in range(world.height)],
            key=lambda t: (t[0] + t[1], t[1])
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
        
        # TODO: make this less call excessively
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
    pygame.display.set_caption(f"Voxel Engine [seed {SEED}]")
    clock = pygame.time.Clock()

    world = build_world(seed=SEED)
    
    texture_engine = TextureEngine(seed=SEED)
    texture_engine.load()

    camera   = Camera(SCREEN_W, SCREEN_H, move_speed=5.0)
    renderer = Renderer(world, camera, texture_engine)
    picker   = Mouse(world, camera)

    viewer = Billboard()
    viewer.init_fonts()

    timeline = Timeline()
    info     = Information(timeline, screen_w=SCREEN_W)
    info.init_fonts()

    print(f'Voxel engine running at {TARGET_FPS}fps')
    
    running = True
    while running:
        dt   = clock.tick(TARGET_FPS) / 1000.0
        keys = pygame.key.get_pressed()
        
        # update camera
        camera.update(dt, keys)
        
        # update simulation
        timeline.tick(dt)
        
        # process input vectors
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEWHEEL:
                # +1 scroll up = zoom in
                camera.zoom(event.y)
                # keep selection valid after zoom (positions shifted)
                if renderer._selected is not None:
                    renderer.set_selection(renderer._selected)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    timeline.toggle_pause()
                elif event.key == pygame.K_ESCAPE:
                    if renderer.inspection_mode:
                        renderer.set_selection(None)
                        viewer.set_selection(None, world)
                    else:
                        running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if renderer.inspection_mode:
                        renderer.set_selection(None)
                        viewer.set_selection(None, world)
                    else:
                        running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    hit = picker.pick(mx, my)
                    if hit is not None:
                        renderer.set_selection(hit)
                        viewer.set_selection(hit, world)

        if camera.moved:
            renderer.set_selection(renderer._selected)

        screen.fill(BG_COLOR)
        
        # commit
        renderer.update(screen)
        viewer.draw(screen, camera)
        info.draw(screen)

        fps = clock.get_fps()
        mode_str = "  [INSPECTION]" if renderer.inspection_mode else "  [VIEWING]"
        pygame.display.set_caption(
            f"Voxel Engine [seed {SEED} | scale {camera.scale:.2f} | {fps:.0f} fps]{mode_str}"
        )
        # next frame
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
