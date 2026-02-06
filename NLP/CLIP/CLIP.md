## CLIP (Contrastive Language–Image Pretraining)

CLIP — это **мультимодальная модель**, обученная для **совместного представления изображений и текста**  
в **едином эмбеддинг-пространстве**.  
Используется для **zero-shot классификации изображений, image–text retrieval,
multimodal search, grounding и других vision-language задач**.

---

### 🔹 Основная идея CLIP

CLIP обучается на **парах (изображение, текст)** с помощью **контрастивного обучения**:

1. **Два энкодера (Dual-Encoder architecture)**  
   - **Image encoder** (ViT или ResNet)  
   - **Text encoder** (Transformer, похожий на BERT, но causal-free)  
   - Каждый энкодер проецирует вход в **общее embedding-пространство** фиксированной размерности.

2. **Контрастивное обучение (Contrastive Learning)**  
   - Для батча из `N` пар модель:
     - максимизирует сходство **правильных пар (image_i, text_i)**
     - минимизирует сходство **неправильных пар**
   - Используется симметричный contrastive loss:
     - image → text
     - text → image

3. **Обучение без явной разметки классов**  
   - CLIP не обучается на фиксированных классах  
   - Текстовые описания задают семантику **на лету**

---

### 🔹 Функция потерь (CLIP Loss)

После нормализации эмбеддингов используется **scaled cosine similarity**:

$$
\text{sim}(I, T) = \frac{I \cdot T}{\|I\| \|T\|}
$$

Логиты:
$$
\text{logits}_{ij} = \frac{\text{sim}(I_i, T_j)}{\tau}
$$

где $$tau$$ — **learnable temperature parameter**.

---

### 🔹 Архитектура

1. **Image Encoder**
   - ResNet или Vision Transformer (ViT)
   - Преобразует изображение в embedding
   - Использует global pooling / CLS token

2. **Text Encoder**
   - Transformer (encoder-only)
   - Использует BPE токенизацию
   - CLS token используется как представление текста

3. **Projection Heads**
   - Линейные слои для приведения эмбеддингов к одной размерности
   - Иногда реализуются как адаптеры

4. **Shared Embedding Space**
   - Изображения и тексты сравниваются напрямую
   - Используется косинусное сходство

---

###  Механизмы работы

1. **Input Processing**
   - Изображения → image encoder
   - Тексты → text encoder

2. **Embedding Normalization**
   - L2-нормализация для стабильного cosine similarity

3. **Similarity Matrix**
   - Каждое изображение сравнивается с каждым текстом в батче

4. **Symmetric Contrastive Optimization**
   - Image → Text
   - Text → Image

---