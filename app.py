from __future__ import annotations
import random
import pygame
import sys

"""
    Config
"""

SEED        = 4
MAP_W       = 16        # tiles wide
MAP_H       = 16        # tiles deep

MAX_HEIGHT  = 6         # max terrain height levels
HIGH_ALT    = 5         # z >= this → top surface
LOW_ALT     = 1         # z <= this → low surface

MAX_SLOPE   = 1         # manhattan decay max neighbour step

TILE_W      = 64        # sprite pixel width
TILE_H      = 64        # sprite pixel height  (includes side faces)
TILE_Z_STEP = 32        # vertical pixels per height unit

SCREEN_W    = 1024
SCREEN_H    = 720
TARGET_FPS  = 60

BG_COLOR    = (0, 0, 0)

SPRITE_COLOR_KEY = (244, 204, 161)

GRASS       = "grass"
DIRT_GRASS  = "dirt_grass"
DIRT        = "dirt"
ROCK        = "rock"

TILE_FILES: dict[str, str] = {
    DIRT:       "assets/tile_dirt.png",
    DIRT_GRASS: "assets/tile_dirt_grass.png",
    GRASS:      "assets/tile_grass.png",
    ROCK:       "assets/tile_rock.png",
}


# ── World Generator ──────────────────────────────────────────────────────

def generate_heightmap(
    width: int  = MAP_W,
    height: int = MAP_H,
    seed: int   = SEED,
    num_peaks: int   = 10,
    spread: float    = 5.0,
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
        if surface_z == HIGH_ALT:
            return DIRT_GRASS
        elif surface_z == MAX_HEIGHT:
            return ROCK
        elif surface_z <= LOW_ALT:
            return DIRT
        else:
            return GRASS
    elif z == surface_z - 1:
        return DIRT_GRASS
    else:
        if z <= 1:
            return ROCK
        return DIRT


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


# ── Sprites ──────────────────────────────────────────────────────

def load_sprites() -> dict[str, pygame.Surface]:
    sprites: dict[str, pygame.Surface] = {}
    for block_type, path in TILE_FILES.items():
        surf = pygame.image.load(path).convert()
        surf.set_colorkey(SPRITE_COLOR_KEY, pygame.RLEACCEL)
        sprites[block_type] = surf
    return sprites


# ── Camera ──────────────────────────────────────────────────────

class Camera:
    """
    Isometric camera with WASD panning.
    Tracks whether it moved this frame via `self.moved` so the
    renderer knows when to invalidate and redraw.
    """

    def __init__(self, screen_w: int, screen_h: int,
                 map_w: int = MAP_W, map_h: int = MAP_H,
                 move_speed: float = 2.0):
        self.screen_w   = screen_w
        self.screen_h   = screen_h
        self.map_w      = map_w
        self.map_h      = map_h
        self.move_speed = move_speed

        # Camera world-space centre position
        self.cam_x: float = 0 #map_w * 0.5
        self.cam_y: float = 0 #map_h * 0.5

        self.tile_w_half    = TILE_W // 2
        self.tile_h_quarter = TILE_H // 4

        # modified state
        self.moved = False

        self._update_offsets()

    def _update_offsets(self) -> None:
        iso_h = (self.map_w + self.map_h) * (TILE_H // 4) + MAX_HEIGHT * TILE_Z_STEP + TILE_H
        self.offset_x = self.screen_w // 2
        self.offset_y = (self.screen_h - iso_h) // 2 + MAX_HEIGHT * TILE_Z_STEP + TILE_H // 2

    def update(self, dt: float, keys) -> None:
        dx, dy = 0.0, 0.0

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += 1.0

        if dx != 0.0 or dy != 0.0:
            if dx != 0.0 and dy != 0.0:
                dx *= 0.7071
                dy *= 0.7071

            self.cam_x += dx * self.move_speed * dt
            self.cam_y += dy * self.move_speed * dt

            padding = 0
            self.cam_x = max(padding, min(self.map_w - padding, self.cam_x))
            self.cam_y = max(padding, min(self.map_h - padding, self.cam_y))
            #print(f' camera x :{self.cam_x} camera y: {self.cam_y}')
            self.moved = True
        else:
            self.moved = False

    def world_to_screen(self, wx: int, wy: int, wz: int) -> tuple[int, int]:
        rel_x = wx - self.cam_x
        rel_y = wy - self.cam_y
        sx = rel_x * self.tile_w_half  - rel_y * self.tile_w_half  + self.offset_x
        sy = rel_x * self.tile_h_quarter + rel_y * self.tile_h_quarter - wz * TILE_Z_STEP + self.offset_y
        return int(sx), int(sy)


# ── Renderer ──────────────────────────────────────────────────────

class Renderer:
    """
    Dirty-flag isometric renderer.

    Static frames: zero tile work — one surface blit only.
    Camera pan:    full redraw triggered by camera.moved flag.
    Tile change:   mark_dirty(x, y) for surgical redraws.
    """

    def __init__(self, world: World, camera: Camera,
                 sprites: dict[str, pygame.Surface]):
        self.world   = world
        self.camera  = camera
        self.sprites = sprites

        self.surface = pygame.Surface((SCREEN_W, SCREEN_H))
        self.surface.fill(BG_COLOR)

        # mark cells for update
        self._dirty: set[tuple[int, int]] = {
            (x, y)
            for x in range(world.width)
            for y in range(world.height)
        }

    def mark_dirty(self, x: int, y: int) -> None:
        self._dirty.add((x, y))

    def _reset(self) -> None:
        for x in range(self.world.width):
            for y in range(self.world.height):
                self._dirty.add((x, y))

    def update(self) -> None:
        if self.camera.moved:
            self._reset()

        if not self._dirty:
            return

        # clear surface
        self.surface.fill(BG_COLOR)

        sorted_cols = sorted(self._dirty, key=lambda t: t[0] + t[1])
        for (x, y) in sorted_cols:
            self._render_column(x, y)

        self._dirty.clear()

    def _render_column(self, x: int, y: int) -> None:
        for z, block_type in self.world.column_blocks(x, y):
            sprite = self.sprites.get(block_type)
            if sprite is None:
                continue
            sx, sy = self.camera.world_to_screen(x, y, z)
            self.surface.blit(sprite, (sx, sy))


# ── Entrypoint ──────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(f"Voxel Engine  —  seed {SEED}")
    clock = pygame.time.Clock()

    world    = build_world(seed=SEED)
    sprites  = load_sprites()
    camera   = Camera(SCREEN_W, SCREEN_H, move_speed=5.0)
    renderer = Renderer(world, camera, sprites)

    print(f"World {world.width}×{world.height}  seed={SEED}")

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