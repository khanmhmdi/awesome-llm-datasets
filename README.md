# Comprehensive Dataset Review Table for LLM Pretraining, SFT and Post-RL Training

A curated collection of open-source datasets for training and fine-tuning Large Language Models, organized by priority and use case.

---

## Tier 1: Essential Core Datasets (Pre-training Foundation)

| Dataset | Size | Tokens | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|--------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **FineWeb** | 44TB | 15T+ | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | ODC-By 1.0 | HuggingFace Hub, datatrove, direct download | State-of-the-art filtered Common Crawl (96 snapshots 2013-2024). Rigorous deduplication and quality filtering. Outperforms C4, Pile, RefinedWeb | **EXCELLENT** - Already optimized for LLM training; translating this gives you production-quality   web text | ⚠️ **Streaming required** - Use `streaming=True` due to size. Install `hf_transfer` for 10x faster download: `pip install huggingface_hub[hf_transfer]` and set `HF_HUB_ENABLE_HF_TRANSFER=1`. 96 CommonCrawl snapshots from 2013-2024. |
| **FineWeb-Edu** | ~5.4TB | 1.3T-5.4T | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-By 1.0 | HuggingFace Hub (samples: 10BT/100BT/350BT) | Educational subset of FineWeb filtered by Llama3-70B classifier. Best for knowledge/reasoning benchmarks | **EXCELLENT** - High educational value; essential for knowledge-intensive   tasks | ⚠️ **Start with sample-100BT** (12GB) to test pipeline. Full dataset is 5.4TB. Educational content has higher quality but narrower domain coverage than general web. |
| **C4** | 750GB | 172B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [allenai/c4](https://huggingface.co/datasets/allenai/c4) | Apache 2.0 | HuggingFace Hub, Google Cloud Storage (`gs://allennep-lm/c4/`) | Classic T5 dataset. Heuristic filtering with NSFW blocklist. Multiple variants: en, en.noclean (6.5TB), realnewslike, multilingual | **VERY GOOD** - Proven track record; pre-cleaned but less filtered than FineWeb | ⚠️ **Heuristic cleaning** means some biases remain. `en.noclean` variant is 6.5TB (unfiltered). Use standard `en` for   translation. Web interface available: [C4 Dataset Explorer](https://c4-search.apps.allenai.org/) |
| **RefinedWeb (Falcon)** | 2.8TB unpacked | 5T | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [tiiuae/falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb) | ODC-By 1.0 | HuggingFace Hub (~500GB compressed) | Stringent filtering from Common Crawl. Released 600B tokens publicly. Outperforms curated corpora. Includes image URLs and alt-text | **EXCELLENT** - Demonstrated that filtered web alone beats curated mixes | ⚠️ **Multimodal-friendly** - contains image URLs and alt-text if you plan vision-language expansion. ~500GB compressed expands to ~2.8TB. Less educational content than FineWeb-Edu. |
| **The Pile** | 825GB | 825B | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [pile.eleuther.ai](https://pile.eleuther.ai/) | Custom (component-dependent) | [The Eye](https://the-eye.eu/) (direct HTTP), no HF mirror | 22 diverse datasets (books, code, web, academic). Variable quality. **Books3 component removed due to copyright** | **MODERATE** - Good diversity but requires heavy filtering; book subset removed | ⚠️ **AVOID Books3 component** - copyright infringement. Variable quality across components. Pre-processed version (SlimPajama) recommended instead. Download from The Eye hosting only. |
| **RedPajama-V1** | ~1TB | 1.2T | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [togethercomputer/RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | Component-dependent | HuggingFace Hub, GitHub (data prep scripts) | Open reproduction of LLaMA data: CommonCrawl (878B), C4 (175B), GitHub (59B), ArXiv (28B), Wikipedia (24B), Books (26B), StackExchange (20B) | **VERY GOOD** - Proven LLaMA-like composition; well-documented processing | ⚠️ **Books component uses Gutenberg** (safe). Pre-deduplicated version (SlimPajama) available that is more efficient. Good for understanding LLaMA-style data composition. |
| **SlimPajama** | ~627GB | 627B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | Apache 2.0 (dataset) + component licenses | HuggingFace Hub (~895GB compressed) | Cleaned, deduplicated RedPajama. 2.5 days processing time on 64 CPU cores. More efficient than original | **VERY GOOD** - Pre-deduplicated saves processing time | ⚠️ **Composition changed**: CommonCrawl reduced from 72.6% to 52.2%, C4 increased from 14.4% to 26.7%. Good if you want less web-heavy mix. |

---

## Tier 2: High-Quality Curated Sources (Knowledge & Academic)

| Dataset | Size | Tokens | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|--------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **Wikipedia (en)** | 19GB | ~3B | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [dumps.wikimedia.org](https://dumps.wikimedia.org/) | CC BY-SA 4.0 / GFDL | XML dumps, HuggingFace (`wikimedia/wikipedia`), Kiwix | Well-curated, structured encyclopedic knowledge. Factual accuracy, neutral tone | **EXCELLENT** - Essential for factual knowledge; structured format translates well | ⚠️ **Raw dumps are 105GB** (cleaned is ~19GB). Use HuggingFace version for easier access: `load_dataset("wikimedia/wikipedia", "20231101.en")`. Rate limit: 3 connections/IP max for dumps. |
| **WikiText-103** | 100M words | ~130M | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [Smerity.com WikiText](https://state.smerity.com/smerity/state/01HRTB51QZMG59MDAX2ME1TCHR) | CC BY-SA 4.0 | Direct download (181 MB), Salesforce Research | Featured/verified good Wikipedia articles. Higher quality than full Wikipedia. Pre-2017 articles only | **EXCELLENT** - Premium quality subset; ideal for high-quality   knowledge base | ⚠️ **Smallest high-quality dataset** (181 MB) - perfect for testing translation pipeline. Only "Good" and "Featured" articles. |
| **Dolma** | ~3TB | 3T | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [allenai/dolma](https://huggingface.co/datasets/allenai/dolma) | AI2 ImpACT License (medium risk) | HuggingFace Hub, GitHub (processing tools) | AllenAI diverse dataset: web, academic, code, books, encyclopedic. High-quality curation. 1,733 languages | **VERY GOOD** - Good diversity; ODC-By license | ⚠️ **Requires accepting AI2 ImpACT License** (medium risk). Contains   content already - use as reference, not translation source. Use streaming for 3TB size. |
| **ArXiv** | ~28B tokens | 28B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv) | arXiv non-exclusive license | Kaggle, Amazon S3 (`s3://arxiv/`), export.arxiv.org | Scientific papers with LaTeX processing. Technical/academic content | **VERY GOOD** - Critical for STEM   capabilities; domain-specific vocabulary | ⚠️ **LaTeX source requires special handling** for   math notation. API rate limit: 4 req/s. Must link back to arXiv for any derived indexes/tools. |
| **S2ORC** | 81.1M papers | ~100B+ | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [AllenAI S2ORC](https://allenai.org/data/s2orc) | Custom (academic use) | Request access via Allen Institute | Semantic Scholar corpus with resolved citations, figures, tables annotations. 8.1M open access full-text | **VERY GOOD** - Rich academic metadata; good for citation understanding | ⚠️ **Requires academic access request** - not instant download. 81.1M papers metadata, 8.1M full-text. Rich citation graph (136M nodes, 467M edges). |
| **OpenWebMath** | 14.7B tokens | 14.7B | ⭐⭐⭐⭐⭐ | ✅ AVAILABLE | [open-web-math/open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math) | Apache 2.0 | HuggingFace Hub | Mathematical web text from Common Crawl with LaTeX preservation. 20x more efficient than general data | **EXCELLENT** - Critical for   mathematical reasoning; unique LaTeX structure | ⚠️ **Essential for STEM  ** - mathematical reasoning is language-agnostic but notation needs   adaptation. Top domains: stackexchange.com (9.55%), nature.com (3.14%), physicsforums.com (2.38%). |

---

## Tier 3: Code & Technical Datasets

| Dataset | Size | Tokens | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|--------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **The Stack v1** | 6.4TB | ~3TB dedup | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [bigcode/the-stack](https://huggingface.co/datasets/bigcode/the-stack) | Permissive licenses (SPDX) | HuggingFace Hub (requires license agreement) | Permissively licensed GitHub repositories. 358 programming languages. Near-deduplicated version available (2.9TB) | **GOOD** - Code is universal, but comments/docs need translation for   code understanding | ⚠️ **Must accept license terms on HuggingFace** (click "Access repository"). Files &gt;1MB excluded. Natural languages in comments: EN, ZH, FR, PT, ES, RU, DE, KO, JA, etc. Use `the-stack-smol` (2.6GB) for testing. |
| **The Stack v2** | 67.5TB | 32.1TB dedup | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [bigcode/the-stack-v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Software Heritage + INRIA agreement | HuggingFace Hub (SWHIDs only), bulk requires email | 600+ programming languages. **Contains SWHIDs only, not actual file content** | **GOOD** - Broader language coverage but complex access | ⚠️ **SWHIDs only** - requires AWS S3 setup to download actual content via `s3://softwareheritage/content/{blob_id}`. Contact datasets@softwareheritage.org for bulk access. 67.5TB full size. |
| **StarCoder Dataset** | 783GB | 250B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) | BigCode OpenRAIL-M v1 | HuggingFace Hub (requires agreement) | The Stack v1 + GitHub issues (54GB), commits (32GB), Jupyter notebooks (13GB). 86 programming languages | **GOOD** - Includes natural language (issues, commits) for conversational code assistance | ⚠️ **Different schemas per component** - load separately: `data_dir="python"` vs `data_dir="github-issues-filtered-structured"`. Loading entire dataset at once may fail. |
| **GitHub Code (BigQuery)** | 59B tokens | 59B | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [BigQuery Public GitHub Data](https://cloud.google.com/bigquery/public-data) | Apache 2.0, BSD, MIT | Google Cloud BigQuery (SQL query access) | Public GitHub repositories filtered by license. Query-based extraction | **MODERATE** - Similar to Stack but more restrictive licensing | ⚠️ **Query costs apply** - first 1TB/month free. Requires Google Cloud account. SQL query interface: `bigquery-public-data.github_repos.sample_contents`. Good for targeted extraction only. |
| **StackExchange** | 20B tokens | 20B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [archive.org/details/stackexchange](https://archive.org/details/stackexchange) | CC BY-SA (varies by date) | Internet Archive (torrent/direct), Academic Torrents | Technical Q&A from 28 largest sites. HTML-cleaned, sorted by score. 7z compressed XML | **VERY GOOD** - High-quality technical dialogue; excellent for   Q&A capabilities | ⚠️ **XML format requires parsing** - 7z compressed. ~54GB download. License varies by date (check license.txt). Pre-processed version in RedPajama (20B tokens) available. |

---

## Tier 4: Instruction & Conversation Datasets (Critical for Chat Capabilities)

## 📊 Ranking Table

| Rank | Dataset | Size | License | Commercial | Quality | Best For |
|:----:|---------|------|---------|:----------:|:-------:|----------|
| **1** | **OpenOrca / FLAN** | 4.2M+ | Various (component-dependent) | ⚠️ Check | ⭐⭐⭐⭐⭐ | Large-scale instruction tuning, reasoning skills, massive pretraining, Massive Scale, Reasoning Explanations Included, FLAN Task Collection Foundation, Proven in Production |
| **2** | **Dolly 2.0 (Databricks)** | 15K | CC BY-SA 3.0 | ✅ Yes | ⭐⭐⭐⭐⭐ | Commercial products, high-quality instruction tuning |
| **3** | **Anthropic HH-RLHF** | 161K conversations | MIT | ✅ Yes | ⭐⭐⭐⭐⭐ | Safety alignment, RLHF/DPO training |
| **4** | **Natural Instructions (AI2)** | 61 tasks / 193K instances | Various (component-dependent) | ⚠️ Check | ⭐⭐⭐⭐⭐ | Cross-task generalization, diverse NLP tasks |
| **5** | **LIMA** | 1K | CC BY-NC-SA (or stricter) | ❌ No | ⭐⭐⭐⭐⭐ | Premium quality over quantity, research |
| **6** | **UltraChat** | ~1.5M | MIT | ✅ Yes | ⭐⭐⭐⭐☆ | Chat models, conversational AI |
| **7** | **ShareGPT** | 90K+ | Unknown (user-contributed) | ❌ Unknown | ⭐⭐⭐⭐☆ | Authentic conversation patterns |
| **8** | **Self-Instruct** | ~52K+ | CC BY-NC 4.0 | ❌ No | ⭐⭐⭐⭐☆ | Research, baseline comparisons |
| **9** | **Alpaca (Stanford)** | 52K | CC BY-NC 4.0 | ❌ No | ⭐⭐⭐⭐☆ | Quick experiments, learning |

---

## Tier 5: Books & Long-Form Narrative

| Dataset | Size | Tokens | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|--------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **Project Gutenberg** | ~70K books | ~10B+ | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [gutenberg.org](https://www.gutenberg.org/) | Public Domain | Website, FTP (`ftp.ibiblio.org`), GitHub mirrors, Kiwix | Public domain books. Classic literature, diverse genres. Pre-1929 publications (US) | **VERY GOOD** - High-quality long-form text; copyright clear | ⚠️ **Language is archaic** (pre-1929). Primarily English, French, German. **No   content**. Use for long-form coherence training. Raw dumps: ~105GB. |
| **PG-19** | Subset | ~5B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [github.com/deepmind/pg19](https://github.com/deepmind/pg19) | Public Domain | GitHub, TensorFlow Datasets, HuggingFace | Gutenberg books published before 1919. Used by LLaMA. 28,602 training, 50 validation, 100 test books | **VERY GOOD** - Pre-filtered, historically validated | ⚠️ **10.94 GiB total**. Pre-1919 only (more archaic than PG). Good for testing long-context (book-level) training. |
| **BookCorpus / Books3** | ~11K books | ~6B | ⭐⭐☆☆☆ | ❌ **NOT AVAILABLE** | Removed from circulation | N/A | N/A | Copyright issues (removed from The Pile). Not recommended | **NOT RECOMMENDED** - Legal risk; avoid | ⚠️ **DO NOT USE** - Copyright infringement issues. Removed from The Pile. Use Gutenberg/PG-19 instead. |
| **FinePDFs** | 3.65TB | 3T | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [HuggingFaceFW/finepdfs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) | ODC-By 1.0 | HuggingFace Hub, datatrove | 475M PDF documents extracted via Docling + RolmOCR. 1,733 languages. **Includes   (fas_Arab)** | **VERY GOOD** - Complements web data; academic/professional content | ⚠️ **Already contains   content** - use as quality reference, NOT for translation. Language code: `fas_Arab` (  in Arabic script). Use datatrove: `ParquetReader("hf://datasets/HuggingFaceFW/finepdfs/data/fas_Arab/train")`. |

---

## Tier 6: Multilingual & Parallel Corpora (Reference Only)

| Dataset | Size | Pairs | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|-------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **ROOTS (BigScience)** | 1.6TB | ~500B | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [bigscience-data/roots](https://huggingface.co/bigscience-data) | Various (component-dependent) | HuggingFace Hub (requires charter acceptance) | 59 languages including  /Arabic script. Ethically sourced with provenance tracking | **REFERENCE ONLY** - Use to validate   quality, not translate from | ⚠️ **Must accept BigScience Data Governance Charter** on HuggingFace. **Contains authentic  ** - use for quality validation only. Do not translate from this. |
| **EuroParl** | 35.6GB | 759M tokens | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [Helsinki-NLP/europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl) | CC-ZERO / Open | HuggingFace Hub, OPUS website | European Parliament proceedings. 21 languages, parallel. 30.32M sentence fragments | **MODERATE** - Formal political language; limited domain | ⚠️ **NO  ** - European languages only. Use for validation/testing multilingual capabilities only. |
| **Tatoeba** | Variable | 50K+ | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [tatoeba.org](https://tatoeba.org) / [manythings.org/anki](https://www.manythings.org/anki/) | CC-BY 2.0 | Website download, tab-delimited files | User-contributed translations. Quality scored by community. 12M+ sentences, 421 languages | **MODERATE** - Good for validation; uneven quality | ⚠️ ** -English pairs available** (~50K estimated). Good for testing translation quality. CC-BY 2.0 (France). Quality varies - community scored. |
| **MASSIVE** | 1M | 1M | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [Amazon Alexa MASSIVE](https://amazon-massive-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz) | CC BY 4.0 | Direct S3 download, HuggingFace | Multilingual NLP tasks across 51 languages. 18 domains (e-commerce, banking, travel) | **MODERATE** - Task-specific; limited general use | ⚠️ **  (fa-IR) included**. Task-specific (intent classification, slot filling). Good for testing   intent understanding, not general LLM training. |
| **UN General Assembly Speeches** | 79 sessions | ~10K speeches | ⭐⭐⭐☆☆ | ✅ AVAILABLE | [digitallibrary.un.org/record/4067189](https://digitallibrary.un.org/record/4067189) | UN Copyright (non-commercial) | UN Digital Library CSV/MD download | Official UN debates and speeches. 1st-78th sessions (1946-2024) | **MODERATE** - Formal diplomatic language; narrow domain | ⚠️ **NO  ** - UN official languages only (Arabic, Chinese, English, French, Russian, Spanish). Non-commercial use with attribution. |

---

## Tier 7: Synthetic & Specialized Datasets

| Dataset | Type | Size | Rating | Download Status | Official Link | License | Access Method | Description | Translation Value | Warnings & Critical Notes |
|---------|------|------|--------|--------------|---------------|---------|---------------|-------------|-----------------|---------------------------|
| **Cosmopedia** | Synthetic | 25B tokens | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [HuggingFaceTB/cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Apache 2.0 | HuggingFace Hub | Synthetic textbooks, blog posts, stories, WikiHow-style. High educational quality. 30M+ files | **VERY GOOD** - Clean structure; easy to translate accurately | ⚠️ **Start with cosmopedia-20k (61MB) or 100k (307MB)** for testing. Full dataset is 25B tokens. Splits: stories, web_samples_v1/v2, wikihow, openstax, stanford, khanacademy. Use `num_proc=12` for faster loading. |
| **OpenAssistant Conversations** | Human | ~100K | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) and [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache 2.0 | HuggingFace Hub | Human-annotated multilingual conversations. 35 languages. Conversation tree structure | **VERY GOOD** - Quality human feedback data | ⚠️ **Only 72   messages** in OASST1. Use as quality reference for conversational  . Project concluded but datasets remain. |
| **UltraFeedback** | Preference | ~300K | ⭐⭐⭐⭐☆ | ✅ AVAILABLE | [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Apache 2.0 | HuggingFace Hub | Preference rankings for RLHF. Multi-response format (~300K prompts) | **VERY GOOD** - For   model alignment | ⚠️ **No   in original**. Translate prompts/responses to create   preference pairs. Use binarized version: `HuggingFaceH4/ultrafeedback_binarized` (~60K pairs). Cleaned version removes TruthfulQA contamination: `argilla/ultrafeedback-binarized-preferences-cleaned`. |

---

## Critical Master Warnings & Recommendations

### 🚫 Datasets to Avoid

| Dataset | Reason |
|---------|--------|
| **BookCorpus / Books3** | Copyright infringement, removed from circulation |
| **Raw Common Crawl** | Use FineWeb/RefinedWeb instead (pre-cleaned) |
| **Pushshift Reddit (raw)** | Quality issues, toxic content, banned subreddits |
| **Twitter/X datasets** | API restrictions, heavy noise, short-form limitations |
| **EuroParl** | No   content (European languages only) |
| **UN General Assembly** | No   content (UN official languages only) |

### ⚠️ Pre-Translation Checklist

1. **Start Small:** Test with WikiText-103 (181MB), LIMA (1K), or Cosmopedia-20k (61MB)
2. **License Check:**
   - ✅ Commercial use: FineWeb, RefinedWeb, Dolly 2.0, HH-RLHF, UltraChat, Cosmopedia
   - ⚠️ Non-commercial: Alpaca, LIMA, Self-Instruct
   - ⚠️ Component-dependent: The Pile, Natural Instructions, OpenOrca
3. **  Validation:** Use ROOTS (  subset), FinePDFs (fas_Arab), Tatoeba (en-fa) as quality references
4. **Streaming Required:** For any dataset &gt;100GB, use `streaming=True` in `load_dataset()`
5. **Post-Translation Deduplication:** Essential - parallel translation creates near-duplicates

### 📊 Recommended   LLM Composition

| Phase | Dataset | Percentage | Priority |
|-------|---------|-----------|----------|
| **Pre-training** | FineWeb or FineWeb-Edu | 50% | Essential |
| | Wikipedia (en) | 15% | Essential |
| | RefinedWeb | 15% | High |
| | ArXiv + OpenWebMath | 10% | High (STEM) |
| | PG-19 / Gutenberg | 10% | Medium |
| **Instruction Tuning** | Natural Instructions | 30% | Essential |
| | Dolly 2.0 | 25% | Essential |
| | HH-RLHF | 20% | Essential (safety) |
| | OpenOrca/FLAN | 25% | High |
| **RLHF Alignment** | UltraFeedback (translated) | 100% | Essential |

### 🔧 Technical Requirements

- **Storage:** Plan for 3-5x compressed size for unpacked data
- **Bandwidth:** Use `hf_transfer` for HuggingFace downloads (10x faster)
- **Memory:** Use `streaming=True` for datasets &gt;50GB
- **Processing:**  -specific validation (RTL text, Arabic script integrity,   characters: پ چ ژ گ)
- **Scale Target:** 100B-300B   tokens minimum (Gemma used 2T-14T total;   effective at smaller scale with high quality)

---

## License

All datasets listed are **freely available** for research use under their respective licenses, with commercial use permitted where specified. Please consult individual dataset licenses before use.

---

## Contributing

This is a living document. Dataset availability and licensing terms may change. Please verify current status via official links before downloading.

---

## Contact

For questions about this dataset compilation, please open an issue in the repository.
