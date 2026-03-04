# Multimodal VLM Agent for Goal-Oriented Navigation

Репозиторий содержит мультимодального RL-агента на основе Vision-Language Model (VLM) для навигации в двумерной grid-среде MiniGrid.

Агент получает изображение состояния среды и текстовый промпт задачи и должен выбрать действие, которое приближает его к цели. В работе исследуется способность таких моделей обобщать стратегию навигации на пространства, размеры которых не встречались во время обучения.

Основные элементы работы:

- обучение через **Supervised Fine-Tuning (SFT)**
- reinforcement learning с использованием **GRPO**
- **curriculum learning** по размерам среды
- стабилизация обучения через **label smoothing**
- вариант **text + action reasoning**, в котором модель сначала описывает среду

---

Важное, из того, что добавлено:
- `dataset.ipynb` - создает датасет на основе shortest-path planner
- `run_eval_env.ipynb` - множественное тестирование чекпоинтов
- `runs.ipynb` - примеры команд для запуска SFT, GRPO и GRPO-lora дообучения text + action через CLI
- `large_empty_grids_test.ipynb` - тестирует большие среды сразу несколько за раз
- `single_test_empty_grid.ipynb` - можно запускать модель для теста

в nanoVLM:
- `train_action_sft.py` - SFT обучение
- `grpo_action_train.py` - GRPO обучение
- `grpo_train_lora.py` - обучение text + action модели с помощью LoRA
- `data/action_collator.py` - сборщик для action
- `models/vision_language_model_action.py` - основная модель для SFT/GRPO обучения
- `models/vl_reasoning_action_model.py` - text + action модель
---

# Установка

```bash
git clone https://github.com/iliailiaai/nanoVLM-action.git
cd nanoVLM-action

pip install torch numpy torchvision pillow datasets huggingface-hub transformers wandb einops minigrid gymnasium
```

Для запуска обучения:  
- Создать датасет через `dataset.ipynb`
- Запустить SFT-обучение `train_action_sft.py`
- Запустить GRPO-обучение `grpo_action_train.py`
- Text + action обучение `grpo_train_lora.py`

Eval оценка: 
- `large_empty_grids_test.ipynb`
- `run_eval_env.ipynb`

Для коротких запусков: 
- `single_test_empty_grid.ipynb`

