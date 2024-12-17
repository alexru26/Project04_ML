# Machine Learning Project 04: Poker AI
This is a program that uses CFR to play Poker (specifically Leduc Hold'em).

## How to run
Make sure you have the necessary packages.

```
pip install -r requirements.txt
```

Run ```main.py``` to use the program.

## Resources Used
- ChatGPT
- https://github.com/datamllab/rlcard
- https://aipokertutorial.com/
- https://orb.binghamton.edu/cgi/viewcontent.cgi?article=1028&context=research_days_posters_2021
- https://github.com/ishikota/PyPokerEngine

## Functionality
- Model 1: Monte Carlos CFR
  - Uses policy table to determine what move to take based on state
  - After traverses down decision tree, goes back up and calculates regret
  - Regret = How much better this choice would've been
    - Higher regret = better choice, lower regret = worse choice
  - Uses regret and action probabilities to calculate gain
  - Update policy table
  - Do that for both players
  - Model trains until reaches Nash Equilibrium
- Model 2: Deep Monte Carlos CFR 
  - Uses two deep neural networks to approximate action probabilities and regret
  - Same procedure as normal MCCFR
- Model 3: Deep Monte Carlos CFR + Bluff
  - Same as deep MCCFR but neural networks take into account another integer that represents likelihood opponent is bluffing

## Reasoning and Issues
- I chose to do Leduc Hold'em instead of No Limit Hold'em because of complexity
  - Leduc Hold'em has much fewer possibilities
    - Probably at most 10<sup>5</sup>
  - No Limit Hold'em has an absurd amount of possibilities
    - ChatGPT says that there are around 10<sup>160</sup> nodes in the total game tree 
    - Other teams burned through multiple GPUS to train their CFR model for NLHE
- Multi-threading during training
  - Useful since I have to run through the decision for both players independently

## Future Improvements
- Add bluff classifier
  - Maybe simulate the rest of game multiple times to see how strong hand is?
- Collect bluff data throughout game and train model
  - Essentially building data of player's face
- Make models faster because can't run normal poker
  - Pruning?
