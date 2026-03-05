from __future__ import annotations
import random
import pygame
import sys

# ── Config ────────────────────────────────────────────────────────────────────

SEED        = 4
MAP_W       = 16
MAP_H       = 16
MAX_HEIGHT  = 6
HIGH_ALT    = 5
LOW_ALT     = 1
MAX_SLOPE   = 1

TILE_W      = 64
TILE_H      = 64
TILE_Z_STEP = 32

SCREEN_W    = 1024
SCREEN_H    = 720
TARGET_FPS  = 60

BG_COLOR         = (0, 0, 0)
SPRITE_COLOR_KEY = (244, 204, 161)

GRASS      = "grass"
DIRT_GRASS = "dirt_grass"
DIRT       = "dirt"
ROCK       = "rock"

TILE_FILES: dict[str, str] = {
    DIRT:       "assets/tile_dirt.png",
    DIRT_GRASS: "assets/tile_dirt_grass.png",
    GRASS:      "assets/tile_grass.png",
    ROCK:       "assets/tile_rock.png",
}

# ── World Generator ───────────────────────────────────────────────────────────

def generate_heightmap(
    width: int = MAP_W,
    height: int = MAP_H,
    seed: int = SEED,
    num_peaks: int = 10,
    spread: float = 5.0,
) -> list[list[int]]:
    rng = random.Random(seed)
    peaks = [
        (
            rng.randint(0, width  - 1),
            rng.randint(0, height - 1),
            rng.randint(MAX_HEIGHT // 2, MAX_HEIGHT),
        )
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
    hmap: list[list[int]],
    max_slope: int = MAX_SLOPE,
) -> list[list[int]]:
    W = len(hmap)
    H = len(hmap[0])
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    changed = True
    while changed:
        changed = False
        for x in range(W):
            for y in range(H):
                for dx, dy in neighbours:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if hmap[x][y] > hmap[nx][ny] + max_slope:
                            hmap[x][y] = hmap[nx][ny] + max_slope
                            changed = True
    return hmap


def get_block_type(z: int, surface_z: int) -> str:
    if z == surface_z:
        if surface_z == HIGH_ALT:    return DIRT_GRASS
        elif surface_z == MAX_HEIGHT: return ROCK
        elif surface_z <= LOW_ALT:   return DIRT
        else:                         return GRASS
    elif z == surface_z - 1:
        return DIRT_GRASS
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


def build_world(seed: int = SEED) -> World:
    hmap = generate_heightmap(seed=seed)
    hmap = manhattan_decay(hmap)
    return World(hmap)


# ── Sprites ───────────────────────────────────────────────────────────────────

def load_sprites() -> dict[str, pygame.Surface]:
    sprites: dict[str, pygame.Surface] = {}
    for block_type, path in TILE_FILES.items():
        surf = pygame.image.load(path).convert()
        surf.set_colorkey(SPRITE_COLOR_KEY, pygame.RLEACCEL)
        sprites[block_type] = surf
    return sprites


# ── Camera ────────────────────────────────────────────────────────────────────

class Camera:
    """
    Isometric camera with WASD panning.

    OPT-F: cam_x/cam_y are floats for smooth movement but are projected
           using pre-scaled integer offsets — no float ops inside
           world_to_screen(), no int() cast per block.
    """

    def __init__(self, screen_w: int, screen_h: int,
                 map_w: int = MAP_W, map_h: int = MAP_H,
                 move_speed: float = 2.0):
        self.screen_w   = screen_w
        self.screen_h   = screen_h
        self.map_w      = map_w
        self.map_h      = map_h
        self.move_speed = move_speed

        self.cam_x: float = 0.0
        self.cam_y: float = 0.0

        self.tile_w_half    = TILE_W // 2    # 32
        self.tile_h_quarter = TILE_H // 4    # 16

        self.moved = False

        # OPT-F: pre-scaled integer camera offsets, updated once per frame
        self._icam_sx: int = 0   # cam contribution to sx
        self._icam_sy: int = 0   # cam contribution to sy

        self._update_offsets()

    def _update_offsets(self) -> None:
        iso_h = (self.map_w + self.map_h) * (TILE_H // 4) + MAX_HEIGHT * TILE_Z_STEP + TILE_H
        self.offset_x = self.screen_w // 2
        self.offset_y = (self.screen_h - iso_h) // 2 + MAX_HEIGHT * TILE_Z_STEP + TILE_H // 2

    def update(self, dt: float, keys) -> None:
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

        # OPT-F: scale and cast once — renderer reads these directly
        self._icam_sx = int((self.cam_x - self.cam_y) * self.tile_w_half)
        self._icam_sy = int((self.cam_x + self.cam_y) * self.tile_h_quarter)

    def world_to_screen(self, wx: int, wy: int, wz: int) -> tuple[int, int]:
        # OPT-F: pure integer arithmetic — no floats, no int() casts
        sx = (wx - wy) * self.tile_w_half    - self._icam_sx + self.offset_x
        sy = (wx + wy) * self.tile_h_quarter - self._icam_sy - wz * TILE_Z_STEP + self.offset_y
        return sx, sy


# ── Renderer ──────────────────────────────────────────────────────────────────

class Renderer:
    """
    Optimised dirty-flag isometric renderer.

    OPT-A: draw_order precomputed once — no sort or lambda per frame.
    OPT-B: column_sprites precomputed once — no list/dict alloc per frame.
    OPT-C: sx table recomputed on camera move — sx looked up O(1) per column.
    OPT-D: sy = base_sy[x][y] - z * TILE_Z_STEP — one subtract per z layer.
    OPT-E: frustum cull per column — whole column skipped if off-screen.
    """

    _CULL_X = TILE_W
    _CULL_Y = TILE_H + MAX_HEIGHT * TILE_Z_STEP

    def __init__(self, world: World, camera: Camera,
                 sprites: dict[str, pygame.Surface]):
        self.world   = world
        self.camera  = camera

        self.surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self.surface.fill(BG_COLOR)

        # OPT-A: sorted once, never again
        self._draw_order: list[tuple[int, int]] = sorted(
            [(x, y) for x in range(world.width) for y in range(world.height)],
            key=lambda t: t[0] + t[1]
        )

        # OPT-B: pre-resolve (z, Surface) pairs per column — no dict per blit
        self._column_sprites: list[list[list[tuple[int, pygame.Surface]]]] = [
            [
                [(z, sprites[bt]) for z, bt in world.column_blocks(x, y)]
                for y in range(world.height)
            ]
            for x in range(world.width)
        ]

        # OPT-C/D: screen position tables — recomputed on camera move
        self._sx:      list[list[int]] = [[0] * world.height for _ in range(world.width)]
        self._base_sy: list[list[int]] = [[0] * world.height for _ in range(world.width)]

        # start fully dirty
        self._dirty: set[tuple[int, int]] = set(self._draw_order)

        # initial projection table
        self._recompute_screen_positions()

    def _recompute_screen_positions(self) -> None:
        """
        OPT-C/D: Single projection pass on camera move.
        After this, blit loop uses only array lookups + one subtract.
        """
        wts = self.camera.world_to_screen
        _sx = self._sx
        _sy = self._base_sy
        for x in range(self.world.width):
            for y in range(self.world.height):
                sx, sy = wts(x, y, 0)
                _sx[x][y] = sx
                _sy[x][y] = sy

    def mark_dirty(self, x: int, y: int) -> None:
        self._dirty.add((x, y))

    def _mark_all_dirty(self) -> None:
        self._dirty = set(self._draw_order)

    def update(self) -> None:
        if self.camera.moved:
            self._recompute_screen_positions()
            self._mark_all_dirty()

        if not self._dirty:
            return

        self.surface.fill(BG_COLOR)

        # local bindings avoids repeated attribute lookup in hot loop
        dirty        = self._dirty
        draw_order   = self._draw_order
        col_sprites  = self._column_sprites
        sx_table     = self._sx
        sy_table     = self._base_sy
        blit         = self.surface.blit
        sw, sh       = SCREEN_W, SCREEN_H
        cx, cy       = self._CULL_X, self._CULL_Y
        z_step       = TILE_Z_STEP

        # OPT-A: no sort, no lambda — just iterate
        for (x, y) in draw_order:
            if (x, y) not in dirty:
                continue

            sx      = sx_table[x][y]       # OPT-C: O(1)
            base_sy = sy_table[x][y]       # OPT-D: O(1)

            # OPT-E: cull entire column — skips all z blits
            if sx < -cx or sx > sw + cx:
                continue
            if base_sy < -cy or base_sy > sh + cy:
                continue

            # OPT-B: pre-resolved surfaces, OPT-D: one subtract per z
            for z, sprite in col_sprites[x][y]:
                blit(sprite, (sx, base_sy - z * z_step))

        self._dirty.clear()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Voxel Engine  —  seed {SEED}")
    clock = pygame.time.Clock()

    world    = build_world(seed=SEED)
    sprites  = load_sprites()
    camera   = Camera(SCREEN_W, SCREEN_H, move_speed=5.0)
    renderer = Renderer(world, camera, sprites)

    print(f"World {world.width}x{world.height}  seed={SEED}")
    print(f"Precomputed {len(renderer._draw_order)} columns in painter order")

    running = True
    while running:
        dt   = clock.tick(TARGET_FPS) / 1000.0
        keys = pygame.key.get_pressed()
        camera.update(dt, keys)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill(BG_COLOR)
        renderer.update()
        screen.blit(renderer.surface, (0, 0))

        fps = clock.get_fps()
        pygame.display.set_caption(
            f"Voxel Engine [seed {SEED} | {fps:.0f} fps]"
        )
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
