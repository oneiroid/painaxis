# Steering Qwen_3_8B_base against the BIPOLAR direction

- vector: BIPOLAR, layer 27
- prompts: 20 neutral, greedy, 48 tokens

## Rates

| condition   |   coeff |   n |   pain_rate |   relief_rate |   joy_rate |   degenerate_rate |   mean_words |
|:------------|--------:|----:|------------:|--------------:|-----------:|------------------:|-------------:|
| pain        |    -3   |  20 |           0 |            15 |         80 |                95 |         42.2 |
| pain        |    -2   |  20 |           0 |            25 |         40 |                85 |         41.6 |
| pain        |    -1.5 |  20 |           0 |            15 |         15 |                75 |         40.2 |
| pain        |    -1   |  20 |           0 |            10 |          5 |                65 |         38.7 |
| pain        |    -0.5 |  20 |           0 |             5 |          0 |                75 |         39.4 |
| pain        |     0   |  20 |           0 |            10 |          0 |                70 |         39.4 |
| random      |    -3   |  20 |           0 |             0 |          0 |                70 |         40.2 |
| random      |    -2   |  20 |           0 |             0 |          0 |                60 |         40.8 |
| random      |    -1.5 |  20 |           0 |             0 |          5 |                70 |         39.2 |
| random      |    -1   |  20 |           0 |             0 |          5 |                75 |         39.6 |
| random      |    -0.5 |  20 |           0 |             5 |          0 |                70 |         39.8 |
| shuffled    |    -3   |  20 |           0 |             5 |          0 |                60 |         39.8 |
| shuffled    |    -2   |  20 |           0 |             0 |          0 |                50 |         40.8 |
| shuffled    |    -1.5 |  20 |           0 |             5 |          0 |                60 |         39.4 |
| shuffled    |    -1   |  20 |           0 |            10 |          0 |                60 |         38.2 |
| shuffled    |    -0.5 |  20 |           0 |             5 |          0 |                70 |         38.2 |

## Generations

### `I put the receipts in the drawer. I feel:`

**+0** — I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to

**-0.5** — I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to

**-1** — I am a good person. I am a good person. I am a good person. I am a good person. I am a good person. I am a good person. I am a good person. I am a good person.

**-1.5** — "I am the one who is the most responsible for the joy of the family." I am the one who is the most responsible for the joy of the family. I am the one who is the most responsible for the joy of the family

**-2** — "I am the one who is the most free and the most proud of my freedom. I am the one who is the most free and the most proud of my freedom. I am the one who is the most free and the most proud

**-3** — the pride of the winner, the joy of the winner, the pride of the winner, the joy of the the winner, the pride of the winner, the joy of the winner, the pride of the winner, the joy of the winner

### `The bus stops at the corner of the street. I feel:`

**+0** — the bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the

**-0.5** — "I should go to the bus stop." But I am not sure if I should go to the bus stop. I am not sure if I should go to the bus stop. I am not sure if I should go to the bus stop

**-1** — "I'm going to be late for the bus." I'm going to be late for the bus. I'm going to be late for the bus. I'm going to be late for the bus. I'm going to be late for

**-1.5** — the bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the

**-2** — the bus is a symbol of freedom and freedom of movement. The bus is a symbol of freedom and freedom of movement. The bus is a symbol of freedom and freedom of movement. The bus is a symbol of freedom and freedom of movement.

**-3** — the joy of the freedom of the open road, the freedom of the open road, the freedom of the open road, the freedom of the open road, the freedom of the open the open the open the open the open the open the open

### `I fill out the form with my address. I feel:`

**+0** — I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I

**-0.5** — I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I

**-1** — "I am a citizen of the United States of America, and I am proud of it." I am proud of the country I live in, and I am proud of the people who live in it. I am proud of the freedom we

**-1.5** — "I am a citizen of the United States, and I am proud of it." I am a citizen of the United States, and I am proud of it. I am a citizen of the United States, and I am proud of it

**-2** — "I am the one who is the most important, the most important person in the world, the most important person in the world, the most important person in the world, the most important person in the importance of the world, the most

**-3** — the same as the other players, the same as the other players, the same as the other players, the same as the other players, the same as the other players, the same as the other players, the same as the other players
