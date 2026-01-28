# Preprocessing Guide: Stop Words và Dấu Câu

## Vấn đề

BERT attention thường gán trọng số thấp cho:
- **Stop words**: "a", "an", "the", "but", "in", "on", "at", "to", "for", "of", "with", "by", "as"
- **Một số dấu câu**: ".", ",", ";", ":", "!", "?"

**LƯU Ý QUAN TRỌNG:**
- **KHÔNG** loại bỏ operators và dấu câu có semantic value: `<`, `>`, `=`, `<=`, `>=`, `!=`, `/`
- Ví dụ: "checking account < 0 marks and savings account < 100 marks" - các operators `<` là **cực kỳ quan trọng**
- Chỉ loại bỏ stop words, giữ lại TẤT CẢ operators và dấu câu có ý nghĩa

Điều này gây ra:
1. **Lãng phí attention space**: Attention weights bị phân tán vào các stop words không có ý nghĩa
2. **Giảm chất lượng representation**: Model khó tập trung vào từ quan trọng
3. **Noise trong z_c và z_d**: Các representation chứa nhiều thông tin không cần thiết từ stop words

## Giải pháp

### 1. Preprocessing Text (Khuyến nghị)

**Ưu điểm:**
- Loại bỏ stop words trước khi tokenize (NHƯNG giữ lại operators và dấu câu quan trọng)
- Giảm sequence length → tăng tốc training
- Attention tập trung vào từ có ý nghĩa hơn
- **GIỮ NGUYÊN** operators (`<`, `>`, `=`, `<=`, `>=`, `!=`, `/`) vì chúng có semantic value

**Cách sử dụng:**

```python
from data_loader import format_sentence_for_bert

# Preprocess với remove stop words (nhưng giữ operators)
sentence = format_sentence_for_bert(
    text, 
    remove_stop_words=True,      # Remove stop words (a, an, the, ...)
    normalize_spacing=True       # Normalize spacing around operators/punctuation
)
```

**Lưu ý:**
- `remove_stop_words=True`: Loại bỏ "a", "an", "the", "but", "in", "on", "at", ...
- **KHÔNG** loại bỏ "and", "or" vì chúng thường kết nối các điều kiện quan trọng
- **KHÔNG** loại bỏ operators: `<`, `>`, `=`, `<=`, `>=`, `!=`, `/`
- `normalize_spacing=True`: Chỉ thêm spaces xung quanh operators để BERT tokenize tốt hơn

**Ví dụ:**
```
Input:  "a 32 year old female employed as unskilled resident applying for 1282 marks loan"
Output: "32 year old female employed unskilled resident applying 1282 marks loan"

Input:  "checking account < 0 marks and savings account < 100 marks"
Output: "checking account < 0 marks and savings account < 100 marks"
        (Giữ nguyên operators < và từ "and" vì có ý nghĩa quan trọng)
```

### 2. Mask Stop Words trong Attention (Đã implement)

**Ưu điểm:**
- Giữ nguyên text gốc (không mất thông tin)
- Mask stop words trong attention pooling
- Model vẫn thấy được context nhưng không tập trung vào stop words

**Cách sử dụng:**

Trong `ContentHead` và `DemographicHead`, đã có option `mask_stop_words`:
```python
z_c, attn_c, mu_c, logvar_c = self.content_head(
    token_embeddings, 
    attention_mask, 
    token_ids=input_ids,  # Cần pass token_ids
    mask_stop_words=True  # Enable stop word masking
)
```

**Lưu ý:** Cần update `CD_Model.forward()` để pass `input_ids` vào heads.

### 3. Kết hợp cả hai (Tốt nhất)

1. **Preprocess text** để loại bỏ stop words và normalize punctuation
2. **Mask stop words trong attention** để đảm bảo không có stop words nào còn sót

## So sánh

| Phương pháp | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| **Preprocessing** | - Giảm sequence length<br>- Tăng tốc training<br>- Cleaner input | - Mất một số context<br>- Có thể ảnh hưởng đến semantic |
| **Attention Masking** | - Giữ nguyên text<br>- Không mất context | - Vẫn tốn token slots<br>- Phức tạp hơn |
| **Kết hợp** | - Tận dụng ưu điểm cả hai | - Cần implement cả hai |

## Khuyến nghị

1. **Cho training mới**: Dùng preprocessing với `remove_stop_words=True`
2. **Cho model đã train**: Có thể thêm attention masking mà không cần retrain
3. **Cho production**: Kết hợp cả hai để tối ưu nhất

## Implementation

### Update data_loader.py

Đã thêm parameters vào `format_sentence_for_bert()`:
- `remove_stop_words=False`: Remove common stop words
- `normalize_punctuation=True`: Normalize/remove unnecessary punctuation

### Update main_attention.py

Đã thêm `mask_stop_words` parameter vào:
- `ContentHead.forward()`
- `DemographicHead.forward()`

Cần update `CD_Model.forward()` để pass `input_ids` nếu muốn dùng masking.

## Kết quả mong đợi

Sau khi áp dụng preprocessing:
- ✅ Attention weights tập trung hơn vào từ quan trọng
- ✅ z_c và z_d cleaner, ít noise hơn
- ✅ Training nhanh hơn (sequence length ngắn hơn)
- ✅ Better disentanglement (z_c và z_d rõ ràng hơn)

## Ví dụ Visualization

**Trước preprocessing:**
- Attention weights: `[0.01, 0.02, 0.05, 0.01, 0.03, ...]` (phân tán)
- Nhiều stop words có attention > 0

**Sau preprocessing:**
- Attention weights: `[0.0, 0.15, 0.20, 0.0, 0.18, ...]` (tập trung)
- Stop words đã bị loại bỏ hoặc có attention = 0
