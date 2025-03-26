
下載dataset
```cmd
!pip install dataset
```
你的 **`pretrain_llm.py`** 需要一個合適的訓練集來進行 LLM 預訓練。以下是幾種不同類型的數據集，取決於你的目標：  

---

## **1️⃣ 開放文本數據集 (適合一般 LLM 預訓練)**
這些數據集包含來自維基百科、書籍和網頁的大量文本，適合用來預訓練 LLM。

| **數據集**  | **描述** | **下載方式** |
|------------|---------|------------|
| **OpenWebText** | 類似 GPT-2 訓練的數據，來自高品質 Reddit 連結 | [Hugging Face](https://huggingface.co/datasets/openwebtext) |
| **Pile** | 800GB 大型文本，包括科學論文、程式碼等 | [Hugging Face](https://huggingface.co/datasets/EleutherAI/the_pile) |
| **WikiText-103** | 英文維基百科內容 | [Hugging Face](https://huggingface.co/datasets/wikitext) |

📌 **範例：下載 OpenWebText 作為訓練集**
```python
from datasets import load_dataset

dataset = load_dataset("openwebtext", split="train")
print(dataset[0])
```

---

## **2️⃣ 專屬領域數據 (適合微調特定應用)**
如果你要訓練 AI 來做特定應用，可以選擇更專精的數據集：

| **應用類型** | **數據集** | **下載方式** |
|-------------|-----------|------------|
| **醫學 LLM** | PubMed Articles | [Hugging Face](https://huggingface.co/datasets/pubmed) |
| **程式碼 LLM** | The Stack (GitHub 開源程式碼) | [Hugging Face](https://huggingface.co/datasets/BigCode/TheStack) |
| **法律 LLM** | Pile-CC Case Law | [Hugging Face](https://huggingface.co/datasets/EleutherAI/the_pile) |

📌 **範例：下載程式碼數據集**
```python
dataset = load_dataset("BigCode/TheStack", split="train")
print(dataset[0])
```

---

## **3️⃣ 自製數據集 (適合個人專案)**
如果你想用自己的數據集來預訓練 LLM，可以使用 **JSON, CSV, 或 TXT** 格式存放你的文本數據。

📌 **JSON 格式**
```json
[
    {"text": "今天的天氣很好，我想出去散步。"},
    {"text": "這是一個人工智慧訓練範例。"}
]
```

📌 **載入自製數據集**
```python
dataset = load_dataset("json", data_files="my_dataset.json")
```

---

## **📌 如何選擇適合的數據集？**
- **如果你要訓練通用 LLM** → **OpenWebText / WikiText**  
- **如果你要讓 LLM 專精某個領域** → **選擇專屬數據 (醫學, 程式碼, 法律)**  
- **如果你有自己的數據 (公司內部, 研究)** → **製作 JSON / CSV 數據集**  

這樣，你的 **`pretrain_llm.py`** 就可以用不同數據集來訓練 LLM 了 🚀
