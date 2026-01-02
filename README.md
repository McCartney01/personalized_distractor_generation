<h1 align = "center">
Tailoring Distractors to Individual Minds: Personalized Distractor Generation via MCTS-Guided Reasoning Reconstruction
</h1>

![](images/method.pdf)

## Getting Started

**1. Installation**
Install `openai` and `math-verify`

```bash
pip install -r requirements.txt
```

**2. Prepare your API key**
We recommend setting the API keys in the environment variables:

```
export OPENAI_API_KEY=your key
```
You can use your own model with the corresponding API key and URL in the OpenAI API form.

**3. Start Running**
```
python dg_new.py --model gpt-3.5-turbo --method mcts --subject eedi
```

## A Walk Through Illustratino of MCTS
![](images/pipeline.pdf)