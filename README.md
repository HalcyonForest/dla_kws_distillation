# dla_kws_distillation
KWS HW
Что я успел попробовать за эту чудесную великолепную и прекрасную неделю выполнения домашки по kws и ничего кроме нее:

1. Knowledge distillation + MSELoss
2. Knowledge distillation + CrossEntropyLoss + Temperature
3. Knowledge distillation + KL_divergence loss + Temperature
4. Knowledge_distillation + Ground truth mixed loss
5. KD + MSELoss + fp16
6. KD + KL_loss + fp16
7. Quantization
8. KD + fp16_training
9. Attention distillation

Что я пробовал в плане архитектуры:
Изначально я не трогал свёртки и страйд и менял лишь размер hidden_size и gru_num_layer. Практика показала, что уменьшать кол-во гру слоёв плохая идея, потому что какие бы я не ставил другие настройки обучение шло плохо и в лучшем случае доходило до 7–6е-5. 

После я решил попробовать что-то делать со свертками, так как это сильно влияет на количество поступаемой информации. 

Я попробовал слегка увеличить страйд и размер ядра. Это позволило улучшить быстродействие и кол-во параметров. 
Оказалось что слишком сильно урезать hidden_size не стоит
Если сокращать кол-во GRU слоев до 1, то модель сходится очень долго. Требуется гораздо больше эпох обучения. 

Конвертация модели в fp16 иногда приводила к резкому ухудшению скора, а иногда нет. Получалось так, что если обучал модель на mse лосс чисто на выводах модели, то получалось не потерять в качестве. Но если обучать на Софтмаксах с температурой, то конвертация в fp16 все резко портила. 

Attention distillation почему-то приводил к тому, что метрика наоборот начинала ухудшаться, видимо где-то накосячил. 
Вот какие графики у меня получились: 

Остальные сравнения канули в лету ввиду того, что у меня в гугл колабе послетали аутпуты ячеек, при переносе на другую почту (квота кончилась)..
Прикрепил самую нормальную модель, которую удалось получить. Получилось неоч потому что только в предпоследний день узнал что изначально hidden_size оказывается 32, а не 64. Так что пришлось все заново переделывать, так как старые модели никуда не годились. Неделя работы насмарку. 

По итогу лучше всего себя показали следующие два метода: KD + fp16 и KD + Temperature

Финальная модель:

@dataclasses.dataclass
```
class StudentConfig:
  keyword: str = 'sheila'
  batch_size: int = 128
  learning_rate: float = 3e-4
  weight_decay: float = 1e-5
  num_epochs: int = 20
  n_mels: int = 40
  cnn_out_channels: int = 3
  kernel_size: Tuple[int, int] = (7,20)
  stride: Tuple[int, int] = (7, 12)
  hidden_size: int = 30
  gru_num_layers: int = 2
  bidirectional: bool = False
  num_classes: int = 2
  sample_rate: int = 16000
  device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
val_loss = 5.500885360278004e-05
После чего, конвертируем ее в fp16.
Memory improvement: 3.7977
Flops_improvement: 4.65 
