# 2048 AI Agent Dashboard

A Streamlit-based dashboard for comparing different AI agents playing the 2048 game. This project implements various AI strategies and provides a visual interface to analyze their performance.

## Features

- Multiple AI agents:
  - Random Agent
  - Greedy Agent
  - Heuristic Agent
  - Expectimax Agent (with configurable depth)
- Interactive dashboard with:
  - Performance metrics
  - Score distributions
  - Max tile distributions
  - Move frequency analysis
  - Performance vs Decision Time visualization
- Configurable simulation settings
- Real-time results visualization

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. In the dashboard:
   - Select the agents you want to compare
   - Choose the number of games to simulate
   - Click "Run Simulation" to start the analysis
   - View the results in various visualizations

## Project Structure

- `streamlit_app.py`: Main dashboard application
- `agents.py`: Implementation of different AI agents
- `game_engine.py`: Core 2048 game logic
- `simulator.py`: Game simulation and statistics collection

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 