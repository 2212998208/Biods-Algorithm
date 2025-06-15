# -*-coding: UTF-8 -*-

import pygame
import numpy as np
import math
import random
import sys

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Boids Algorithm - Flocking Simulation with Parameter Control")

# Color definitions
BACKGROUND = (10, 10, 30)
BOID_COLOR = (100, 200, 255)
NEIGHBOR_COLOR = (255, 100, 100, 100)
TEXT_COLOR = (200, 220, 255)
RANGE_COLOR = (30, 100, 150, 50)
ARROW_COLOR = (100, 200, 255)
HIGHLIGHT_COLOR = (255, 215, 0)
SLIDER_BG = (40, 40, 60)
SLIDER_FG = (80, 150, 220)
SLIDER_HANDLE = (200, 230, 255)

# Default Boids parameters
params = {
    "max_speed": 4.0,
    "perception_radius": 32,
    "separation_weight": 3.0,
    "alignment_weight": 2.3,
    "cohesion_weight": 1.73,
    "edge_force": 7.25,
    "separation_radius": 30
}

# Parameter ranges
param_ranges = {
    "max_speed": (0.5, 10.0),
    "perception_radius": (10, 200),
    "separation_weight": (0.1, 5.0),
    "alignment_weight": (0.1, 5.0),
    "cohesion_weight": (0.1, 5.0),
    "edge_force": (0.1, 15.0),
    "separation_radius": (10, 100)
}

# Font
font = pygame.font.SysFont("cambriamath", 24)
title_font = pygame.font.SysFont("cambriamath", 36)


class Boid:
    def __init__(self, x, y):
        self.position = np.array([float(x), float(y)])
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * params["max_speed"]
        self.acceleration = np.array([0.0, 0.0])
        self.size = 6
        self.neighbors = []

    def update(self):
        # Update velocity
        self.velocity += self.acceleration
        # Limit maximum speed
        speed = np.linalg.norm(self.velocity)
        if speed > params["max_speed"]:
            self.velocity = (self.velocity / speed) * params["max_speed"]

        # Update position
        self.position += self.velocity
        self.acceleration = np.array([0.0, 0.0])

        # Boundary handling
        self.handle_boundaries()

    def apply_force(self, force):
        self.acceleration += force

    def handle_boundaries(self):
        margin = 50  # Fixed margin for simplicity
        turn_factor = params["edge_force"]

        if self.position[0] < margin:
            self.velocity[0] += turn_factor
        if self.position[0] > WIDTH - margin:
            self.velocity[0] -= turn_factor
        if self.position[1] < margin:
            self.velocity[1] += turn_factor
        if self.position[1] > HEIGHT - margin:
            self.velocity[1] -= turn_factor

    def flock(self, boids):
        self.neighbors = []
        separation = np.array([0.0, 0.0])
        alignment = np.array([0.0, 0.0])
        cohesion = np.array([0.0, 0.0])

        total_separation = 0
        total_alignment = 0
        total_cohesion = 0

        for other in boids:
            if other is self:
                continue

            distance = np.linalg.norm(self.position - other.position)

            if distance < params["perception_radius"]:
                self.neighbors.append(other)

                # Separation: avoid crowding
                if distance < params["separation_radius"]:
                    diff = self.position - other.position
                    diff = diff / (distance * distance + 1e-5)  # Avoid division by zero
                    separation += diff
                    total_separation += 1

                # Alignment: steer toward average heading
                alignment += other.velocity
                total_alignment += 1

                # Cohesion: steer toward average position
                cohesion += other.position
                total_cohesion += 1

        # Apply separation rule
        if total_separation > 0:
            separation = separation / total_separation
            separation = self.normalize(separation) * params["max_speed"]
            separation -= self.velocity
            separation = self.limit(separation, 0.2)  # Fixed max force
            separation *= params["separation_weight"]
            self.apply_force(separation)

        # Apply alignment rule
        if total_alignment > 0:
            alignment = alignment / total_alignment
            alignment = self.normalize(alignment) * params["max_speed"]
            alignment -= self.velocity
            alignment = self.limit(alignment, 0.2)  # Fixed max force
            alignment *= params["alignment_weight"]
            self.apply_force(alignment)

        # Apply cohesion rule
        if total_cohesion > 0:
            cohesion = cohesion / total_cohesion
            cohesion -= self.position
            cohesion = self.normalize(cohesion) * params["max_speed"]
            cohesion -= self.velocity
            cohesion = self.limit(cohesion, 0.2)  # Fixed max force
            cohesion *= params["cohesion_weight"]
            self.apply_force(cohesion)

    def normalize(self, vector):
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector

    def limit(self, vector, max_value):
        norm = np.linalg.norm(vector)
        if norm > max_value:
            return (vector / norm) * max_value
        return vector

    def draw(self, screen):
        # Draw perception range
        pygame.draw.circle(screen, RANGE_COLOR, self.position.astype(int),
                           params["perception_radius"], 1)

        # Draw neighbor connections
        for neighbor in self.neighbors:
            pygame.draw.line(screen, NEIGHBOR_COLOR,
                             self.position.astype(int),
                             neighbor.position.astype(int), 1)

        # Draw arrow
        angle = math.atan2(self.velocity[1], self.velocity[0])
        length = 15

        # Arrow tip
        end_point = (
            self.position[0] + math.cos(angle) * length,
            self.position[1] + math.sin(angle) * length
        )

        # Arrow side points
        arrow_left = (
            end_point[0] - math.cos(angle - math.pi / 6) * length / 2,
            end_point[1] - math.sin(angle - math.pi / 6) * length / 2
        )

        arrow_right = (
            end_point[0] - math.cos(angle + math.pi / 6) * length / 2,
            end_point[1] - math.sin(angle + math.pi / 6) * length / 2
        )

        # Draw arrow polygon
        pygame.draw.polygon(screen, ARROW_COLOR, [
            end_point,
            arrow_left,
            self.position,
            arrow_right
        ])

        # Draw arrow body
        pygame.draw.line(screen, ARROW_COLOR,
                         self.position.astype(int),
                         end_point, 2)


def create_boids(num=None):
    if num is None:
        return []
    return [Boid(random.randint(50, WIDTH - 50),
                 random.randint(50, HEIGHT - 50)) for _ in range(num)]


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.dragging = False

        # Calculate handle position
        self.handle_radius = 10
        self.handle_x = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width

    def draw(self, screen):
        # Draw slider background
        pygame.draw.rect(screen, SLIDER_BG, self.rect)

        # Draw filled portion
        fill_width = (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_width, self.rect.height)
        pygame.draw.rect(screen, SLIDER_FG, fill_rect)

        # Draw handle
        handle_pos = (self.handle_x, self.rect.y + self.rect.height // 2)
        pygame.draw.circle(screen, SLIDER_HANDLE, handle_pos, self.handle_radius)

        # Draw label and value
        label_text = font.render(f"{self.label}: {self.value:.2f}", True, TEXT_COLOR)
        screen.blit(label_text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                handle_pos = (self.handle_x, self.rect.y + self.rect.height // 2)
                if (mouse_pos[0] - handle_pos[0]) ** 2 + (mouse_pos[1] - handle_pos[1]) ** 2 <= self.handle_radius ** 2:
                    self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                mouse_x, _ = pygame.mouse.get_pos()
                # Constrain to slider bounds
                mouse_x = max(self.rect.x, min(mouse_x, self.rect.x + self.rect.width))
                self.handle_x = mouse_x
                # Calculate new value
                self.value = self.min_val + (mouse_x - self.rect.x) / self.rect.width * (self.max_val - self.min_val)
                return True
        return False


def draw_info(screen):
    # Display title
    title = title_font.render("Boids Algorithm - Flocking Simulation", True, TEXT_COLOR)
    screen.blit(title, (20, 15))

    # Display instructions
    instructions = [
        "Warning: !!!Please switch your input method to English!!!",
        "Controls:",
        "- Click and drag sliders to adjust parameters",
        "- Click to add boids",
        "- Space: Add random boid",
        "- C: Clear all boids",
        "- R: Reset parameters",
        "- V: Toggle visuals",
        "- ESC: Quit"
    ]

    for i, text in enumerate(instructions):
        text_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(text_surface, (WIDTH - 600, 60 + i * 30))


def main(params=None):
    boids = create_boids(None)
    clock = pygame.time.Clock()

    # Create sliders for parameters
    sliders = [
        Slider(20, 100, 300, 15, param_ranges["max_speed"][0], param_ranges["max_speed"][1],
               params["max_speed"], "Max Speed"),
        Slider(20, 160, 300, 15, param_ranges["perception_radius"][0], param_ranges["perception_radius"][1],
               params["perception_radius"], "Perception Radius"),
        Slider(20, 220, 300, 15, param_ranges["separation_weight"][0], param_ranges["separation_weight"][1],
               params["separation_weight"], "Separation Weight"),
        Slider(20, 280, 300, 15, param_ranges["alignment_weight"][0], param_ranges["alignment_weight"][1],
               params["alignment_weight"], "Alignment Weight"),
        Slider(20, 340, 300, 15, param_ranges["cohesion_weight"][0], param_ranges["cohesion_weight"][1],
               params["cohesion_weight"], "Cohesion Weight"),
        Slider(20, 400, 300, 15, param_ranges["edge_force"][0], param_ranges["edge_force"][1],
               params["edge_force"], "Edge Force"),
        Slider(20, 460, 300, 15, param_ranges["separation_radius"][0], param_ranges["separation_radius"][1],
               params["separation_radius"], "Separation Radius")
    ]

    # Add some initial randomness
    for boid in boids:
        angle = random.uniform(0, 2 * math.pi)
        boid.velocity = np.array([math.cos(angle), math.sin(angle)]) * params["max_speed"]

    running = True
    show_visuals = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Space to add new boid
                    boids.append(Boid(random.randint(50, WIDTH - 50),
                                      random.randint(50, HEIGHT - 50)))
                elif event.key == pygame.K_v:
                    # Toggle visuals
                    show_visuals = not show_visuals
                elif event.key == pygame.K_c:
                    # Clear all boids
                    boids = []
                elif event.key == pygame.K_r:
                    # Reset parameters
                    params = {
                        "max_speed": 4.0,
                        "perception_radius": 32,
                        "separation_weight": 3.0,
                        "alignment_weight": 2.3,
                        "cohesion_weight": 1.73,
                        "edge_force": 7.25,
                        "separation_radius": 30
                    }
                    # Reset slider values
                    for i, slider in enumerate(sliders):
                        slider.value = list(params.values())[i]
                        slider.handle_x = slider.rect.x + (slider.value - slider.min_val) / (
                                    slider.max_val - slider.min_val) * slider.rect.width
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Mouse click to add boid
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    # Only add if not clicking on slider area
                    if y > 500 or x > 350:
                        boids.append(Boid(x, y))

            # Handle slider events
            param_changed = False
            for slider in sliders:
                if slider.handle_event(event):
                    param_changed = True

            # Update params if any slider changed
            if param_changed:
                params["max_speed"] = sliders[0].value
                params["perception_radius"] = sliders[1].value
                params["separation_weight"] = sliders[2].value
                params["alignment_weight"] = sliders[3].value
                params["cohesion_weight"] = sliders[4].value
                params["edge_force"] = sliders[5].value
                params["separation_radius"] = sliders[6].value

        # Fill background
        screen.fill(BACKGROUND)

        # Update and draw boids
        for boid in boids:
            boid.flock(boids)
            boid.update()
            if show_visuals:
                boid.draw(screen)
            else:
                # Draw simplified arrow without visuals
                angle = math.atan2(boid.velocity[1], boid.velocity[0])
                length = 15
                end_point = (
                    boid.position[0] + math.cos(angle) * length,
                    boid.position[1] + math.sin(angle) * length
                )
                pygame.draw.line(screen, ARROW_COLOR,
                                 boid.position.astype(int),
                                 end_point, 2)
                pygame.draw.circle(screen, ARROW_COLOR, boid.position.astype(int), 3)

        # Draw sliders
        for slider in sliders:
            slider.draw(screen)

        # Draw info panel
        draw_info(screen)

        # Draw boid count
        count_text = font.render(f"Boids: {len(boids)}", True, TEXT_COLOR)
        screen.blit(count_text, (20, HEIGHT - 40))

        # Draw toggle hint
        hint = font.render("Press 'V' to toggle visuals", True, TEXT_COLOR)
        screen.blit(hint, (WIDTH // 2 - 100, HEIGHT - 40))

        # Update display
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main(params)
