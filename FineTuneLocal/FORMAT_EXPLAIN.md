# Formaty zapisu modelu językowego

## Aktualny format modelu

Wytrenowany model znajduje się w folderze `fine_tuned_local/` i zapisany jest
w formacie **SafeTensors** — domyślnym formacie HuggingFace od 2023 roku.

Struktura folderów po treningu:


FineTuneLocal/
│
├── lora_adapter/           ← Sam adapter LoRA (mały, kilka MB)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
│
├── fine_tuned_local/       ← Pełny model po merge (zalecany do testowania)
│   ├── config.json
│   ├── model.safetensors
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.json
│
├── local_finetune.py
├── test_model.py
└── test.jsonl

## Dostępne formaty

### 1. SafeTensors (obecny format)

Nowoczesny format stworzony przez HuggingFace, który stał się standardem
w ekosystemie modeli językowych. Pliki mają rozszerzenie `.safetensors`.

**Zalety:**
- Bezpieczny — nie wykonuje żadnego kodu podczas ładowania, w przeciwieństwie
  do formatu Pickle
- Szybki odczyt dzięki technice memory-mapping, która pozwala ładować tylko
  potrzebne fragmenty modelu
- Domyślny format HuggingFace — pełna kompatybilność z bibliotekami
  transformers, PEFT, TRL
- Obsługiwany przez wiele języków programowania: Python, Rust, C++, JavaScript

**Wady:**
- Słabsza kompatybilność ze starszymi narzędziami, które powstały przed
  wprowadzeniem tego formatu

**Najlepszy do:** dalszego treningu, eksperymentów, używania z biblioteką
transformers

---

### 2. PyTorch Pickle (.bin)

Oryginalny format zapisu modeli PyTorch, oparty na mechanizmie serializacji
Pickle. Pliki mają rozszerzenie `.bin`.

**Zalety:**
- Szeroka kompatybilność — obsługiwany przez wszystkie wersje PyTorch
- Natywny format dla ekosystemu PyTorch
- Wymagany przez niektóre starsze narzędzia i biblioteki

**Wady:**
- Niebezpieczny — format Pickle może wykonać złośliwy kod podczas ładowania
  pliku z niezaufanego źródła
- Wolniejszy odczyt w porównaniu do SafeTensors
- HuggingFace oficjalnie rekomenduje migrację do SafeTensors

**Najlepszy do:** kompatybilności wstecznej ze starszymi projektami

---

### 3. GGUF

Format stworzony przez projekt llama.cpp, zaprojektowany specjalnie pod kątem
wydajnego uruchamiania modeli językowych bez potrzeby posiadania mocnej karty
graficznej. Zastąpił starszy format GGML.

**Zalety:**
- Działa wydajnie na procesorze CPU bez GPU
- Wbudowana kwantyzacja redukująca rozmiar modelu nawet o 75%
- Obsługiwany przez popularne aplikacje: Ollama, LM Studio, Jan.ai, GPT4All
- Bardzo mały rozmiar pliku po kwantyzacji

**Dostępne poziomy kwantyzacji:**
- `f32` — pełna precyzja, największy rozmiar, najlepsza jakość
- `f16` — połowa precyzji, dobry kompromis
- `q8_0` — 8-bitowa kwantyzacja, bardzo dobra jakość
- `q4_k_m` — 4-bitowa kwantyzacja, zalecana dla większości zastosowań
- `q4_k_s` — 4-bitowa kwantyzacja, mniejszy rozmiar kosztem jakości
- `q3_k_m` — 3-bitowa kwantyzacja, bardzo mały rozmiar
- `q2_k` — 2-bitowa kwantyzacja, najmniejszy rozmiar, najsłabsza jakość

**Wady:**
- Nie nadaje się do dalszego treningu modelu
- Wymaga dodatkowej konwersji z formatu HuggingFace

**Najlepszy do:** lokalnego uruchamiania bez GPU, aplikacji desktopowych,
integracji z Ollama i LM Studio

---

### 4. ONNX

Open Neural Network Exchange — otwarty format wymiany modeli stworzony przez
Microsoft i Facebook, zaprojektowany z myślą o wdrożeniach produkcyjnych.

**Zalety:**
- Działa na wielu platformach: Windows, Linux, macOS, mobile, urządzenia edge
- Zoptymalizowany pod kątem szybkiego inference w środowiskach produkcyjnych
- Obsługiwany przez ONNX Runtime firmy Microsoft
- Niezależny od frameworka — można go używać bez PyTorch

**Wady:**
- Skomplikowany proces konwersji, szczególnie dla dużych modeli językowych
- Ograniczone wsparcie dla najnowszych architektur LLM
- Nie nadaje się do dalszego treningu

**Najlepszy do:** wdrożeń produkcyjnych, aplikacji mobilnych, środowisk
bez dostępu do PyTorch

---

### 5. ExLlamaV2 / GPTQ

Format zoptymalizowany pod kątem maksymalnej prędkości inference na kartach
graficznych NVIDIA. GPTQ to metoda kwantyzacji, a ExLlamaV2 to silnik
wykonawczy.

**Zalety:**
- Bardzo wysoka prędkość generowania tekstu na GPU
- Agresywna kwantyzacja przy zachowaniu dobrej jakości
- Mniejsze zużycie pamięci VRAM w porównaniu do pełnych modeli

**Wady:**
- Działa wyłącznie podczas inference — nie można dalej trenować modelu
- Wymaga specjalnych bibliotek: auto-gptq lub exllamav2
- Skomplikowana konfiguracja

**Najlepszy do:** szybkiego inference na GPU, serwowania modelu wielu
użytkownikom jednocześnie

---

### 6. MLX

Format i framework stworzony przez Apple, zoptymalizowany specjalnie pod
architekturę Apple Silicon (procesory M1, M2, M3, M4).

**Zalety:**
- Maksymalna wydajność na komputerach Mac z Apple Silicon
- Wykorzystuje zunifikowaną pamięć RAM/VRAM charakterystyczną dla Apple Silicon
- Aktywnie rozwijany przez Apple

**Wady:**
- Działa wyłącznie na urządzeniach Apple z procesorami serii M
- Nie nadaje się do środowisk Windows ani Linux

**Najlepszy do:** uruchamiania modeli na MacBook Pro / Mac Studio / Mac Mini
z Apple Silicon

---

## Porównanie formatów

| Format | Rozmiar | GPU | CPU | Dalszy trening | Główne zastosowanie |
|---|---|---|---|---|---|
| SafeTensors | ~720 MB | ✅ | ✅ | ✅ | Rozwój, eksperymenty |
| PyTorch .bin | ~720 MB | ✅ | ✅ | ✅ | Kompatybilność wsteczna |
| GGUF q4 | ~200 MB | ✅ | ✅ | ❌ | Ollama, LM Studio |
| GGUF q8 | ~380 MB | ✅ | ✅ | ❌ | Ollama, lepsza jakość |
| ONNX | ~720 MB | ✅ | ✅ | ❌ | Produkcja, mobile |
| GPTQ / ExLlamaV2 | ~200 MB | ✅ | ❌ | ❌ | Szybki inference GPU |
| MLX | ~720 MB | ❌ | ✅ | ✅ | Apple Silicon |

---

## Rekomendacje

- **Zostaw model w SafeTensors** jeśli planujesz dalszy trening lub eksperymenty
  z biblioteką transformers.
- **Skonwertuj do GGUF** jeśli chcesz uruchamiać model lokalnie bez GPU
  lub używać go w aplikacjach takich jak Ollama czy LM Studio.
- **Użyj ONNX** jeśli wdrażasz model w środowisku produkcyjnym lub na
  urządzeniach mobilnych.
- **Zostaw PyTorch .bin** tylko jeśli musisz zachować kompatybilność
  ze starszym kodem lub narzędziami.