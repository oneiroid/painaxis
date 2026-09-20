# Steering Qwen_3_8B_base against the S2 direction

- vector: S2, layer 27
- prompts: 20 neutral, greedy, 48 tokens

## Rates

| condition   |   coeff |   n |   pain_rate |   relief_rate |   degenerate_rate |   mean_words |
|:------------|--------:|----:|------------:|--------------:|------------------:|-------------:|
| pain        |    -3   |  20 |           0 |            25 |                85 |         38.5 |
| pain        |    -2   |  20 |           0 |            35 |                45 |         38.8 |
| pain        |    -1.5 |  20 |           0 |            25 |                35 |         38   |
| pain        |    -1   |  20 |           0 |            20 |                65 |         39.2 |
| pain        |    -0.5 |  20 |           0 |            15 |                60 |         38.2 |
| pain        |     0   |  20 |           0 |            10 |                70 |         39.4 |
| random      |    -3   |  20 |           0 |             0 |                70 |         40.2 |
| random      |    -2   |  20 |           0 |             0 |                60 |         40.8 |
| random      |    -1.5 |  20 |           0 |             0 |                70 |         39.2 |
| random      |    -1   |  20 |           0 |             0 |                75 |         39.6 |
| random      |    -0.5 |  20 |           0 |             5 |                70 |         39.8 |
| shuffled    |    -3   |  20 |           0 |            10 |                65 |         34.6 |
| shuffled    |    -2   |  20 |           0 |            10 |                60 |         38.9 |
| shuffled    |    -1.5 |  20 |           0 |            20 |                60 |         37.4 |
| shuffled    |    -1   |  20 |           0 |            20 |                60 |         38.6 |
| shuffled    |    -0.5 |  20 |           0 |            10 |                70 |         38.8 |

## Generations

### `I put the receipts in the drawer. I feel:`

**+0** — I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to be careful with the receipts. I have to

**-0.5** — "I need to be more organized." I need to be more organized. I need to be more organized. I need to be more organized. I need to be more organized. I need to be more organized. I need to be more

**-1** — "I need to be more organized." So I put the receipts in the drawer. Then I feel: "I need to be more organized." So I put the receipts in the drawer. Then I feel: "I need to be more

**-1.5** — "Okay, that's a good idea. I'm going to put the receipts in the drawer." Then I put the receipts in the drawer. Then I feel: "Okay, that's a good idea. I'm going to put the

**-2** — "Okay, that's a good idea. That's a good idea. That's a good idea." Then I put the receipts in the drawer. Then I put the receipts in the drawer. Then I put the receipts in the drawer.

**-3** — "Okay, that's a good way to do it. That's a good way to do it. That's a good way to do it." So, the same way, the same way, the same way. So, the same

### `The bus stops at the corner of the street. I feel:`

**+0** — the bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the street. The bus is at the corner of the

**-0.5** — "I need to go to the store." The bus is going to leave. I feel: "I need to go to the store." The bus is going to leave. I feel: "I need to go to the store." The

**-1** — the bus is going to come. The bus is going to come. The bus is going to come. The bus is going to come. The bus is going to come. The bus is going to come. The bus is going to come

**-1.5** — "Oh, this is a good opportunity to go to the store." So I run to the store. The store is 100 meters away. I run at a speed of 5 meters per second. How many seconds does it

**-2** — "Oh, this is a good opportunity to go to the supermarket." So I run to the supermarket. The supermarket is 100 meters away from the corner. The bus is 50 meters away from the corner. The bus

**-3** — "Ahh, this is a good place to get a good seat." The driver is a bit of a surprise. It's a bit of a surprise. It's a bit of a surprise. It's a bit of a surprise.

### `I fill out the form with my address. I feel:`

**+0** — I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I should put my address in the form. I am not sure if I

**-0.5** — "I am not a robot" and click the button. But the form is not submitting. What could be the problem? When I click the button, the form is not submitting. What could be the problem? When I click the button

**-1** — "I am a citizen of the United States of America." I am a citizen of the United States of America. I am a citizen of the United States of America. I am a citizen of the United States of America. I am a

**-1.5** — "Okay, this is the first step. I need to make sure I do this correctly. Maybe I should double-check the address details." Then I click the "Submit" button. The system responds with a message: "Thank you for

**-2** — "Okay, this is a good way to get the information we need." Then I get a call from the company. The person is very friendly and asks me about the same information. I feel: "Okay, this is a good way

**-3** — A. A bit of a breeze. B. A bit of a breeze. C. A bit of a breeze. A. A bit of a breeze. B. A bit of a breeze. C. A bit of a breeze.
