# 🧠 LifeKI – Kooperative Überlebenssimulation mit Reinforcement Learning

![LifeKI Simulation](./docs/lifeki_demo.gif)

> **LifeKI** ist eine interaktive Python-Simulation, in der mehrere Agenten mithilfe von **Q-Learning** lernen, gemeinsam zu überleben.  
> Jeder Agent muss Nahrung finden, Energie verwalten und kooperativ agieren, um als Gemeinschaft länger zu überleben.  
> Mit jedem Simulationslauf entwickelt sich das Verhalten der KI weiter – und du kannst es live im **Overlay-Graphen** verfolgen.

---

## 🚀 Features
- **Kooperatives Überleben**: Mehrere KI-Agenten mit gemeinsamer Ressourcenstrategie  
- **Reinforcement Learning**: Q-Learning mit dynamischer Exploration (ε-greedy)  
- **Fortschrittsspeicherung**: Q-Tabelle & Highscores bleiben über Sessions hinweg bestehen  
- **Live-Visualisierung**: Echtzeit-Graph für Überlebenszeit & Geburtenrate  
- **Individuelles Verhalten**: Jeder Agent lernt, Entscheidungen situationsabhängig zu treffen  
- **Schöne Visualisierung** mit Pygame-Gitter, Agenten-Icons und Energieanzeige  

---

## 🛠 Installation

> **Empfohlen:** Python **3.9** oder höher

### 1️⃣ Repository klonen
```bash
git clone https://github.com/nlblime/LifeKI.git
cd LifeKI
