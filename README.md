# Local Privacy-Preserving Transcript Redactor

A 100% offline and local Python tool for redacting Personally Identifiable Information (PII) from meeting transcripts, text files, and PDFs.

## Features

- **100% Offline & Private**: Zero data transmission, zero external API calls, and zero telemetry. All NLP and pattern matching run directly on your local CPU/RAM.
- **Multi-Format Support**:
  - WebVTT transcript files (`.vtt`)
  - PDF documents (`.pdf`)
  - Plain text files (`.txt`)
  - Direct string input via CLI
- **Smart PII Detection**:
  - Names (Person names & kinship relations like "my mother Sarah")
  - Student IDs, Campus IDs, NetIDs, Badges, and general ID numbers
  - Email addresses, phone numbers, IP addresses, dates, and locations
- **Customizable Output**:
  - Configurable redaction placeholders (`[NAME_REDACTED]`, `[ID_REDACTED]`, etc.)
  - Audit logging flag (`--audit`) to inspect detected entity types and confidence scores

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BraddyM25/transcript-redaction-tool.git
   cd transcript-redaction-tool
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SpaCy language model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### 1. Redact a WebVTT transcript
```bash
python redact.py --vtt path/to/transcript.vtt
```
By default, this outputs `path/to/transcript_redacted.vtt`. You can specify a custom output path with `--output <path>`.

### 2. Redact a PDF transcript
```bash
python redact.py --pdf path/to/document.pdf --output redacted_doc.txt
```

### 3. Redact a Plain Text file
```bash
python redact.py --txt path/to/file.txt --output redacted.txt
```

### 4. Redact Direct Text Input
```bash
python redact.py --text "My name is John Doe and my NetID is jd123." --audit
```

### 5. Running the Built-in Test Suite
```bash
python redact.py --test
```
