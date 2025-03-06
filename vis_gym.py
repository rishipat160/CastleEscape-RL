import pygame
import sys
import time
import random
from mdp_gym import CastleEscapeEnv  # Import the CastleEscapeMDP class


WIDTH, HEIGHT = 600, 840  # 5x5 grid, each room is 120x120 pixels
GRID_SIZE = 5
CELL_SIZE = WIDTH // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
YELLOW = (255, 255, 0)  # Color for the goal room

## Please add the file paths and use them to render some cool looking stuff.
IMGFILEPATH = {

}

# Setup display
screen=None

game_ended = False
action_results = [None, None, None, None, None]

fps = 60
sleeptime = 0.1
clock = None

# Initialize MDP game
game = CastleEscapeEnv()

# Initialize Pygame
def setup(GUI=True):
    global screen
    if GUI:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Castle Escape MDP Visualization")
        # Constants

# Map room to grid cell positions
def position_to_grid(position):
    row, col = position
    return col * CELL_SIZE, row * CELL_SIZE

# Draw the grid for the rooms and shade console area
def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, 600, CELL_SIZE):
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Shade
    rect = pygame.Rect(0, 600, WIDTH, HEIGHT - 600)
    pygame.draw.rect(screen, GRAY, rect)

# Draw the goal room in yellow
def draw_goal_room():
    x, y = position_to_grid(game.goal_room)
    rect = pygame.Rect(x, y, CELL_SIZE-2, CELL_SIZE-2)
    pygame.draw.rect(screen, YELLOW, rect)
    font = pygame.font.Font(None, 36)
    label = font.render('Goal', True, BLACK)
    screen.blit(label, (x + CELL_SIZE // 4 +1, y + CELL_SIZE // 4 +1))

# Draw player at a given position
def draw_player(position):
    x, y = position_to_grid(position)
    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2
    pygame.draw.circle(screen, GREEN, (center_x, center_y), CELL_SIZE // 4)

# Draw guards at their positions
def draw_guards(guard_positions):
    for guard, position in guard_positions.items():
        x, y = position_to_grid(position)
        rect = pygame.Rect(x + CELL_SIZE // 4, y + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2)
        pygame.draw.rect(screen, RED, rect)
        # Label the guard
        font = pygame.font.Font(None, 24)
        label = font.render(guard, True, WHITE)
        screen.blit(label, (x + CELL_SIZE // 4, y + CELL_SIZE // 4))

# Draw player and guard together if they are in the same room
def draw_player_and_guard_together(position, guard_positions):
    guards_in_room = [guard for guard, pos in guard_positions.items() if pos == position]
    guards_not_in_room = [guard for guard in guard_positions if guard not in guards_in_room]
    if guards_in_room:
        x, y = position_to_grid(position)
        # Draw the player
        player_x = x + CELL_SIZE // 4
        player_y = y + CELL_SIZE // 2
        pygame.draw.circle(screen, GREEN, (player_x, player_y), CELL_SIZE // 6)
        
        # Draw the guard
        guard_x = x + 3 * CELL_SIZE // 4
        guard_y = y + CELL_SIZE // 2
        pygame.draw.rect(screen, RED, (guard_x - CELL_SIZE // 8, guard_y - CELL_SIZE // 8, CELL_SIZE // 4, CELL_SIZE // 4))
        
        # Label the guard
        font = pygame.font.Font(None, 24)
        label = font.render(guards_in_room[0], True, WHITE)
        screen.blit(label, (guard_x - 10, guard_y - 10))

    for guard in guards_not_in_room:
        draw_guards({guard: guard_positions[guard]})

# Draw player health status
def draw_health(health):
    font = pygame.font.Font(None, 36)
    health_text = f"Health: {health}"
    health_surface = font.render(health_text, True, BLUE)
    screen.blit(health_surface, (10, HEIGHT - 40))

# Display victory or defeat message
def display_end_message(message):
    font = pygame.font.Font(None, 100)
    text_surface = font.render(message, True, DARK_GRAY)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)

# Main loop
def main():
    global game_ended, action_results
    clock = pygame.time.Clock()
    running = True
    end_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and not game_ended:
                    action = "UP"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
                if event.key == pygame.K_s and not game_ended:
                    action = "DOWN"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
                if event.key == pygame.K_a and not game_ended:
                    action = "LEFT"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
                if event.key == pygame.K_d and not game_ended:
                    action = "RIGHT"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
                if event.key == pygame.K_f and not game_ended:
                    action = "FIGHT"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
                if event.key == pygame.K_h and not game_ended:
                    action = "HIDE"
                    result = game.step(action)
                    action_results.append(f"Action: {action}, Result: {result}")
        screen.fill(WHITE)
        draw_grid()

        # Draw the goal room
        draw_goal_room()

        # Check if player and a guard are in the same room and draw them together
        if game.current_state['player_position'] in game.current_state['guard_positions'].values():
            draw_player_and_guard_together(game.current_state['player_position'], game.current_state['guard_positions'])
        else:
            # Draw the player and guards in separate positions
            draw_player(game.current_state['player_position'])
            draw_guards(game.current_state['guard_positions'])

        # Display player health
        draw_health(game.current_state['player_health'])

        # Check for terminal state
        if game.is_terminal() == 'goal':
            game_ended = True
            end_message = "Victory!"
        elif game.is_terminal() == 'defeat':
            game_ended = True
            end_message = "Defeat!"

        if game_ended:
            display_end_message(end_message)
            game_ended = False

        # Print the latest 5 results on the screen
        font = pygame.font.Font(None, 30)
        console_surface = font.render("Console", True, BLUE)
        screen.blit(console_surface, (10, 610))
        font = pygame.font.Font(None, 24)
        y_offset = 645
        for result in action_results[-5:]:
            result_surface = font.render(result, True, BLACK)
            screen.blit(result_surface, (10, y_offset))
            y_offset += 30

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

def refresh(obs, reward, done, info, delay=0.1):
    global fps, sleeptime, game_ended, clock, action_results, game
    
    try:
        action = info['action']
    except:
        action = "None"

    result = "Pos: {}, Health: {}, Guard In Cell: {}, Reward: {}, Action: {}".format(obs['player_position'], game.int_to_health_state[obs['player_health']], obs['guard_in_cell'], reward, action)

    if None in action_results:
        action_results[action_results.index(None)] = result
    else:
        action_results.pop(0)
        action_results.append(result)

    fps = 60
    clock = pygame.time.Clock()
    screen.fill(WHITE)
    draw_grid()

    # Draw the goal room
    draw_goal_room()

    # Check if player and a guard are in the same room and draw them together
    if game.current_state['player_position'] in game.current_state['guard_positions'].values():
        draw_player_and_guard_together(game.current_state['player_position'], game.current_state['guard_positions'])
    else:
        # Draw the player and guards in separate positions
        draw_player(game.current_state['player_position'])
        draw_guards(game.current_state['guard_positions'])

    # Display player health
    draw_health(game.current_state['player_health'])

    if game.is_terminal() == 'goal':
        game_ended = True
        end_message = "Victory!"
    elif game.is_terminal() == 'defeat':
        game_ended = True
        end_message = "Defeat!"

    if game_ended:
        display_end_message(end_message)
        game_ended = False

    # Print the latest 5 results on the screen
    font = pygame.font.Font(None, 30)
    console_surface = font.render("Console", True, BLUE)
    screen.blit(console_surface, (10, 610))
    font = pygame.font.Font(None, 24)
    y_offset = 645
    for result in action_results:
        result_surface = font.render(result, True, BLACK)
        screen.blit(result_surface, (10, y_offset))
        y_offset += 30

    # Check for terminal state
    pygame.display.flip()
    clock.tick(fps)
    time.sleep(sleeptime)


if __name__ == "__main__":
    setup()
    main()