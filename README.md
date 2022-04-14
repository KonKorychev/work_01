# Поиск подозрительных транзакций
Построить алгоритм поиска «подозрительных» транзакций клиентов, которые тратили бонусные баллы со своих пластиковых карт в магазинах. Оценка качества алгоритма - площадь под ROC-кривой (AUC-ROC).

**Входные данные:**
- **TRANSACTION_ID** - идентификатор транзакции
- **IS_FRAUD** - метка «подозрительной операции»
- **TRANSACTION_DATE_IN** - дата транзакции в числовом формате
- **TRANSACTION_AMOUNT** - объем транзакции в валюте
- **PRODUCT_TYPE** - тип продукта в покупке
- **PAY_SYSTEM** - тип системы оплаты
- **CARD_TYPE** - тип пластиковой карты
- **PURCHASER_EMAIL_DOMAIN** - домен отправителя средств
- **RECIPIENT_EMAIL_DOMAIN** - домен получателя средств
- **SYSTEM_VERSION** - версия операционной системы устройства
- **BROWSER_TYPE** - тип браузера на устройстве
- **SCREEN_PARAMS** - разрешение экрана устройства
- **DEVICE_TYPE** - тип устройства
- **DEVICE_INFO** - информация об устройстве

```python
# Импорт основных библиотек
import numpy as np
import pandas as pd
import re

# Импорт библиотеки научных расчетов
import scipy.stats as sps

# Импорт библиотек машинного обучения
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Импорт библиотеки построения диаграмм и графиков
from matplotlib import pyplot as plt
import seaborn as sns

# Указание режима отображения диаграмм
%matplotlib inline

# Установка начального значения генератора случайных чисел
np.random.seed(42)
```

## Загрузка исходных данных
```python
# Загрузка исходных данных по транзакциям клиентов
data_classification = pd.read_csv('data_classification_train.csv')

# Вывод загруженных данных
data_classification.head()
```
![png](Images/output_20_0.png)

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

