import pygame
import random
import sys
import json
import os
import subprocess
from collections import defaultdict

# --- Einstellungen ---
GRID_W, GRID_H     = 10, 10
CELL_SIZE          = 50
FPS                = 60
WINDOW_HEIGHT      = GRID_H * CELL_SIZE + 300
WINDOW_SIZE        = (GRID_W * CELL_SIZE, WINDOW_HEIGHT)

# --- Spiel-Parameter ---
MOVE_COST          = 1
FOOD_ENERGY        = 30
NEEDY_THRESHOLD    = 50
BIRTH_THRESHOLD    = 100
FOOD_COUNT         = 30
INITIAL_AGENTS     = 3
COMMUNITY_REWARD   = 5

# --- RL-Parameter ---
EPSILON_START      = 0.3
MIN_EPSILON        = 0.01
ALPHA               = 0.6
EPSILON_DECAY       = 0.995
GAMMA               = 0.9

# --- Farben ---
BG_COLOR           = (30, 30, 30)
GRID_COLOR         = (70, 70, 70)
FOOD_COLOR         = (200, 200, 50)
AGENT_COLORS       = [
    (100, 250, 100),
    (250, 100, 100),
    (100, 100, 250),
    (250, 250, 100),
    (100, 250, 250),
]

# --- Pfade ---
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
QTABLE_FILE  = os.path.join(SCRIPT_DIR, "qtable.json")
HIGHSCORE_FILE = os.path.join(SCRIPT_DIR, "highscores.json")

def load_qtable():
    if os.path.exists(QTABLE_FILE):
        with open(QTABLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    data = {}
    with open(QTABLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return data

def save_qtable(qtable):
    with open(QTABLE_FILE, "w", encoding="utf-8") as f:
        json.dump(qtable, f, indent=4)

def load_highscores():
    if os.path.exists(HIGHSCORE_FILE):
        with open(HIGHSCORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"max_steps": 0, "max_births": 0, "max_q": 0.0}

def save_highscores(highscores):
    with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
        json.dump(highscores, f, indent=4)

# Initiales Laden
q_table = load_qtable()
highscores = load_highscores()
last_simulation = {"steps": 0, "births": 0, "max_q": 0.0}

# --- Environment ---
class Environment:
    def __init__(self, width, height, food_count):
        self.width, self.height = width, height
        self.max_food = food_count
        self.regrow_prob = 0.1
        self.spawn_food(food_count)

    def spawn_food(self, count):
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(coords)
        self.food = set(coords[:count])

    def regrow(self):
        if random.random() < self.regrow_prob and len(self.food) < self.max_food:
            empties = [
                (x, y) for x in range(self.width) for y in range(self.height)
                if (x, y) not in self.food
            ]
            if empties:
                self.food.add(random.choice(empties))

    def has_food(self, x, y):
        return (x, y) in self.food

    def remove_food(self, x, y):
        self.food.discard((x, y))

# --- Agent ---
class Agent:
    ACTIONS = ['up', 'down', 'left', 'right', 'stay']

    def __init__(self, name, color, q_table):
        self.name = name
        self.color = color
        self.x = random.randrange(GRID_W)
        self.y = random.randrange(GRID_H)
        self.energy = 100
        self.q_table = q_table
        self.epsilon = EPSILON_START
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.last_state = None
        self.last_action = None

    def is_alive(self):
        return self.energy > 0

    def get_state_key(self, agents, env):
        energy_bin = min(self.energy // 10, 9)
        hungry = sum(
            1 for o in agents if o is not self and o.is_alive() and o.energy < NEEDY_THRESHOLD
        )
        hungry_bin = min(hungry, 4)
        dists = [abs(self.x - fx) + abs(self.y - fy) for fx, fy in env.food] or [GRID_W + GRID_H]
        dist_bin = min(dists[0] // 2, 5)
        return f"{energy_bin},{hungry_bin},{dist_bin}"

    def choose_action(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.ACTIONS)
        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)
        qs = self.q_table[state_key]
        max_idx = qs.index(max(qs))
        return self.ACTIONS[max_idx]

    def update_q(self, state_key, action, reward, next_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * len(self.ACTIONS)
        if next_key not in self.q_table:
            self.q_table[next_key] = [0.0] * len(self.ACTIONS)
        idx = self.ACTIONS.index(action)
        old_q = self.q_table[state_key][idx]
        next_max = max(self.q_table[next_key])
        self.q_table[state_key][idx] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)

    def act(self, env, agents):
        state_key = self.get_state_key(agents, env)
        action = self.choose_action(state_key)
        self.last_state = state_key
        self.last_action = action

        ate = False
        if action == 'stay' and env.has_food(self.x, self.y):
            env.remove_food(self.x, self.y)
            self.energy += FOOD_ENERGY
            ate = True
        elif action != 'stay':
            dx, dy = {
                'up': (0, -1), 'down': (0, 1),
                'left': (-1, 0), 'right': (1, 0)
            }[action]
            self.x = max(0, min(GRID_W - 1, self.x + dx))
            self.y = max(0, min(GRID_H - 1, self.y + dy))
            self.energy -= MOVE_COST

        reward = 0
        if len(agents) == INITIAL_AGENTS:
            reward += COMMUNITY_REWARD
        reward += 1
        if ate:
            reward += 20
        if self.energy >= NEEDY_THRESHOLD:
            reward += 5
        else:
            reward -= 5
        reward -= sum(1 for o in agents if o is not self and o.is_alive() and o.energy < NEEDY_THRESHOLD)
        if len(agents) == 1:
            reward -= 50

        next_key = self.get_state_key(agents, env)
        self.update_q(state_key, action, reward, next_key)

        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Kooperative Überlebensstrategie")
font = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()

subprocess.Popen([sys.executable, os.path.join(SCRIPT_DIR, "show_graph.py")])
outer_active = True
while outer_active:
    agent_counters = defaultdict(int)
    env = Environment(GRID_W, GRID_H, FOOD_COUNT)
    agents = []
    for i in range(INITIAL_AGENTS):
        letter = chr(ord('A') + i)
        agent_counters[letter] += 1
        name = f"{letter}{agent_counters[letter]}"
        agents.append(Agent(name, AGENT_COLORS[i % len(AGENT_COLORS)], q_table))
    births = 0
    step = 0
    running = True

    while running:
        clock.tick(FPS)
        step += 1

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                outer_active = False

        for ag in list(agents):
            if ag.is_alive():
                ag.act(env, agents)

        env.regrow()
        before = {id(a) for a in agents if a.is_alive()}
        agents = [a for a in agents if a.is_alive()]
        died = before - {id(a) for a in agents}
        if died:
            for ag in agents:
                if ag.last_state and ag.last_action:
                    ag.update_q(ag.last_state, ag.last_action, -10 * len(died), ag.last_state)

        newborns = []
        for ag in agents:
            if ag.energy > BIRTH_THRESHOLD:
                letter = ag.name[0]
                agent_counters[letter] += 1
                child_name = f"{letter}{agent_counters[letter]}"
                child = Agent(child_name, ag.color, q_table)
                child.x, child.y = ag.x, ag.y
                child.energy = ag.energy // 2
                ag.energy //= 2
                births += 1
                newborns.append(child)
        agents.extend(newborns)

        if not agents:
            running = False

        screen.fill(BG_COLOR)
        for x in range(GRID_W):
            for y in range(GRID_H):
                pygame.draw.rect(screen, GRID_COLOR, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)
        for (x, y) in env.food:
            r = pygame.Rect(x * CELL_SIZE + 10, y * CELL_SIZE + 10, CELL_SIZE - 20, CELL_SIZE - 20)
            pygame.draw.rect(screen, FOOD_COLOR, r)
        for ag in agents:
            cx = ag.x * CELL_SIZE + CELL_SIZE // 2
            cy = ag.y * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(screen, ag.color, (cx, cy), CELL_SIZE // 3)
            screen.blit(font.render(ag.name, True, (0, 0, 0)), (cx - 10, cy - 10))

        ui_y = GRID_H * CELL_SIZE
        pygame.draw.rect(screen, BG_COLOR, (0, ui_y, WINDOW_SIZE[0], WINDOW_HEIGHT - ui_y))
        q_size = len(q_table)
        avg_maxq = sum(max(vals) for vals in q_table.values()) / q_size if q_size else 0.0

        info = [
            f"Schritt: {step}",
            f"Agenten: {len(agents)}  Geburten: {births}",
            f"Zustände: {q_size}, Ø max-Q: {avg_maxq:.2f}",
            f"🏆 Längste Zeit: {highscores['max_steps']} (zuletzt: {last_simulation['steps']})",
            f"🏆 Meiste Geburten: {highscores['max_births']} (zuletzt: {last_simulation['births']})"
        ]
        for ag in agents:
            info.append(f"{ag.name}: E={ag.energy} Bin={min(ag.energy//10,9)}")
        for i, line in enumerate(info):
            screen.blit(font.render(line, True, (230, 230, 230)), (10, ui_y + 10 + i * 26))

        pygame.display.flip()

    # Nach Ende der Runde: Werte speichern
    highscores["max_steps"] = max(highscores["max_steps"], step)
    highscores["max_births"] = max(highscores["max_births"], births)
    highscores["max_q"] = max(highscores["max_q"], avg_maxq)
    last_simulation["steps"] = step
    last_simulation["births"] = births
    last_simulation["max_q"] = avg_maxq
    # Werte in history.json speichern
    HISTORY_FILE = os.path.join(SCRIPT_DIR, "history.json")
    def append_to_history(entry):
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    history = []
        history.append(entry)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    append_to_history({
        "steps": step,
        "births": births,
        "max_q": avg_maxq
    })
    save_qtable(q_table)
    save_highscores(highscores)
