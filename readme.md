# Genetic Algorithms with Jetpack Joyride

## How to Run

- Install dependencies with `pip install -r requirements.txt`
- Run `python main.py` to start the program in training mode to see what training looks like
  - If you get to a genetic model you like and want to save, press `Space` or `w`, which will save the model weights in a `beast.pkl` file
  - We've included a `beast.pkl` file in the repo which has good performance on the game
- If you want to load the model weights to see how it does, run `python main.py load` to load weights from `beast.pkl`
- Scores from the most recent round of training are plotted and saved in `scores.png`