## MoE (Mixture-of-Experts)

MoE — это **Transformer-архитектура с несколькими экспертами**, предназначенная для **масштабирования моделей без пропорционального увеличения вычислений**.  
Используется для **языкового моделирования, генерации текста и других NLP задач**, где требуется очень большая модель с эффективным использованием ресурсов.

---

### 🔹 Основная идея MoE

MoE добавляет в стандартный Transformer **несколько параллельных Feed-Forward экспертов** в каждом слое:

1. **Expert layers (Feed-Forward Experts)**  
   - Каждый слой содержит **несколько FFN-экспертов**.  
   - Для каждого токена выбирается один или несколько экспертов через **routing/gating**.  
   - Эксперты вычисляются **только для выбранных токенов**, что экономит память.

2. **Gating / Routing**  
   - **Гейт** решает, какой эксперт будет обрабатывать каждый токен:  
     $$
     p_i = \text{softmax}(W_\text{gate} \cdot h_i)
     $$  
     где $h_i$ — скрытое состояние токена $i$, а $p_i$ — распределение по экспертам.  
   - Токен может пойти к **top-1** или **top-2** экспертам.  

3. **Load balancing / Drop tokens**  
   - Чтобы распределение токенов было равномерным, используют **load balancing loss**.  
   - Если эксперт переполнен (capacity), лишние токены могут быть **dropped**, что предотвращает перегрузку.  

---

### 🔹 Применение

- Масштабирование языковых моделей (до сотен миллиардов параметров)  
- Генерация текста и диалоговые системы  
- Машинный перевод и суммаризация  
- Специализация экспертов под разные типы токенов или задачи  

---

### 🔹 Механизмы работы

1. **Input Embedding + Positional Encoding**  
   - Токены преобразуются в векторные представления и добавляются позиционные эмбеддинги.  

2. **Multi-Head Self-Attention**  
   - Каждое токенное состояние может видеть все предыдущие токены (autoregressive) или всю последовательность (encoder MoE).  

3. **Mixture-of-Experts Feed-Forward**  
   - Каждый токен проходит через **top-k экспертов**, выбранных гейтом.  
   - Эксперты вычисляют свои FFN только для выделенных токенов.  

4. **Gating / Routing**  
   - Softmax по экспертам → выбирается top-k  
   - Токены маршрутизируются и умножаются на **routing weights** (важность каждого эксперта для токена)  

5. **Residual Connections + LayerNorm**  
   - Стабилизируют обучение и ускоряют сходимость.  

6. **Load balancing**  
   - Дополнительный loss для равномерного использования всех экспертов:  
     $$
     \text{loss}_\text{balance} = \text{Coef} \cdot \sum_j \left(\frac{\text{tokens}_j}{\text{tokens}_\text{total}} - \frac{1}{N_\text{experts}}\right)^2
     $$  

7. **Output Projection / Decoding**  
   - После MoE слой → LayerNorm → линейный слой → логиты для словаря.

---

### 🔹 Архитектурные особенности

- **Multiple Feed-Forward Experts per layer**  
- **Sparse computation** — вычисляются только выбранные эксперты  
- **Routing weights** — вес каждого эксперта для токена  
- **Optional top-k routing** (top-1, top-2)  
- **Load balancing / Drop tokens** для равномерной загрузки  
- **Stack of MoE Transformer blocks** с residual connections  

---

### ✅ Преимущества MoE

- Масштабируемость до **сотен миллиардов параметров** без линейного роста вычислений  
- Эксперты могут **специализироваться** под разные токены или задачи  
- Позволяет **обучать огромные модели на стандартных GPU кластерах**  
- Sparse activation → экономия памяти и FLOPs  

---

### ⚠️ Недостатки

- Сложнее реализовать и оптимизировать  
- Routing может стать **узким местом**, особенно при top-1 без load balancing  
- Неравномерная загрузка экспертов → “dead experts”  
- Drop tokens может уменьшать качество генерации при малых batch size  

---

### 🔹 Популярные MoE модели

| Model | Type | Description |
|-------|------|-------------|
| **Switch Transformer** | Decoder-only | Google, Top-1/Top-2 routing, до 1.6T параметров |
| **GShard** | Encoder-decoder | Google, масштабируемый MoE для перевода и NLP |
| **BASE Layer MoE** | Decoder | Используется в Bloom, Jurassic и других LLM |
| **Mixture-of-Experts GPT** | Decoder-only | Похож на GPT, но с MoE в FFN слоях |

---

### 🔹 Сравнение с обычным GPT

| Feature | GPT | GPT-MoE |
|---------|-----|---------|
| FFN | Dense | Sparse experts |
| Computation | Full | Only selected experts |
| Parameter efficiency | Linear | High (sparse activation) |
| Token specialization | None | Experts can specialize |
| Scaling | Expensive | Efficient for huge models |

---

