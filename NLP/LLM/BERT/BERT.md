## BERT (Bidirectional Encoder Representations from Transformers)

BERT — это **bidirectional Transformer модель**, предназначенная для **представления текста и понимания контекста**.  
Используется для **классификации текста, NER, вопрос-ответ, парсинга и других NLP-задач**.

---

### 🔹 Основная идея BERT

BERT представляет собой **stack двунаправленных трансформеров (encoder-only)**:

1. **Encoder-only architecture — bidirectional Transformer**  
   - Каждый токен видит **контекст с обеих сторон** (слева и справа).  
   - В отличие от autoregressive моделей, BERT не предсказывает токен один за другим.  
   - На вход подаются токены и позиционные эмбеддинги.  

2. **Masked Language Modeling (MLM)**  
   - Во время pretraining случайные токены **маскируются**.  
   - Модель учится предсказывать **замаскированные токены**, используя контекст слева и справа.  

3. **Next Sentence Prediction (NSP)** (опционально, в оригинальном BERT)  
   - Модель учится предсказывать, является ли второе предложение продолжением первого.  

4. **Pretraining**  
   - MLM loss:  
     $$
     \mathcal{L}_{MLM} = - \sum_{t \in M} \log P(x_t | x_{\text{context}})
     $$  
     где \(M\) — позиции замаскированных токенов.  
   - NSP loss (если используется): бинарная классификация для пары предложений.

---

### 🔹 Применение

- Классификация текста (sentiment, intent)  
- Named Entity Recognition (NER)  
- Вопрос-ответ (SQuAD, QA системы)  
- Понимание отношений между предложениями  
- Файнтюнинг под downstream задачи  

---

### 🔹 Механизмы работы

1. **Input Embedding + Positional Encoding + Segment Embedding**  
   - Токены преобразуются в векторы.  
   - Добавляются позиционные эмбеддинги.  
   - Добавляются сегментные эмбеддинги (для пары предложений).

2. **Bidirectional Multi-Head Self-Attention**  
   - Каждый токен видит **контекст слева и справа**.  
   - Нет causal mask, используется только padding mask для игнорирования паддингов.

3. **Feed-Forward Network (FFN)**  
   - Применяется отдельно к каждому токену после attention.  

4. **Layer Normalization и Residual Connections**  
   - Стабилизируют обучение и ускоряют сходимость.

5. **Output Heads**  
   - **MLM head**: предсказание замаскированных токенов.  
   - **Classification head**: для задач классификации с использованием [CLS] токена.  

---

### 🔹 Архитектурные особенности

- **Encoder-only Transformer**  
- **Bidirectional attention** для понимания контекста  
- **Stack of Transformer blocks**: Multi-head Attention + FFN + LayerNorm + residual connections  
- **Pretraining** на MLM (и опционально NSP)  
- **Flexible sequence length** благодаря позиционным эмбеддингам  
- **[CLS] token** используется для классификационных задач  

---

### ✅ Преимущества BERT

- Полный двунаправленный контекст  
- Отлично подходит для понимания текста и извлечения информации  
- Легко адаптируется под downstream задачи через fine-tuning  
- Сильные результаты на широком спектре NLP задач  

---

### ⚠️ Недостатки

- Требует много памяти и вычислительных ресурсов, особенно при больших версиях (BERT-large)  
- Невозможна генерация текста токен за токеном (не autoregressive)  
- Длинные последовательности требуют оптимизаций при inference
