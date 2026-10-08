"""
Local Privacy-Preserving Transcript Redactor
-------------------------------------------
100% Offline & Local Processing.
Zero data transmission, zero cloud API calls, zero model training.
All NLP inference runs entirely on your local CPU and RAM.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Optional, Tuple

# pyrefly: ignore [missing-import]
import pypdf
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig




class CustomIdRecognizer(EntityRecognizer):
    """
    Detects Student IDs, Campus IDs, Employee IDs, NetIDs, and General ID numbers
    without capturing surrounding non-ID words.
    """
    def __init__(self):
        super().__init__(
            supported_entities=["STUDENT_ID", "ID_NUMBER"],
            name="custom_id_recognizer"
        )
        # Direct pattern with strict boundaries
        self.direct_pattern = re.compile(
            r"(?i)\b(?:net\s*id|netid|student\s*id|campus\s*id|employee\s*id|badge\s*#?|\bid\b(?:\s*(?:number|#|code))?)\s*(?:is|:|=|#|-)?\s*([a-z0-9_-]{3,15})\b"
        )
        # Contextual NetID pattern (e.g., asking for NetID across conversational turns)
        self.context_netid_pattern = re.compile(
            r"(?i)(?:net\s*id|netid)[\s\S]{1,60}?\b([a-z]{2,4}\d{1,5})\b"
        )

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        target = "STUDENT_ID" if "STUDENT_ID" in entities else "ID_NUMBER"
        for match in self.direct_pattern.finditer(text):
            results.append(
                RecognizerResult(
                    entity_type=target,
                    start=match.start(1),
                    end=match.end(1),
                    score=0.95
                )
            )
        for match in self.context_netid_pattern.finditer(text):
            results.append(
                RecognizerResult(
                    entity_type=target,
                    start=match.start(1),
                    end=match.end(1),
                    score=0.98
                )
            )
        return results


class KinshipNameRecognizer(EntityRecognizer):
    """
    Detects names of family members and personal relations in conversation:
    e.g. 'my mother Sarah', 'my brother David', 'my roommate Alex'.
    """
    def __init__(self):
        super().__init__(
            supported_entities=["PERSON"],
            name="kinship_name_recognizer"
        )
        self.pattern = re.compile(
            r"(?i)\b(?:my\s+(?:mother|mom|father|dad|brother|sister|son|daughter|uncle|aunt|cousin|wife|husband|roommate|partner|friend))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        )

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results = []
        if "PERSON" not in entities:
            return results
        for match in self.pattern.finditer(text):
            results.append(
                RecognizerResult(
                    entity_type="PERSON",
                    start=match.start(1),
                    end=match.end(1),
                    score=0.90
                )
            )
        return results




class LocalRedactor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initializes the Presidio engine with local SpaCy NLP pipeline.
        Everything runs locally on CPU/RAM.
        """
        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": spacy_model}]
        }
        provider = NlpEngineProvider(nlp_configuration=config)
        self.analyzer = AnalyzerEngine(
            nlp_engine=provider.create_engine(),
            supported_languages=["en"]
        )
        self.anonymizer = AnonymizerEngine()

        # Register custom local recognizers
        self.analyzer.registry.add_recognizer(CustomIdRecognizer())
        self.analyzer.registry.add_recognizer(KinshipNameRecognizer())

        # Default entity types to look for
        self.default_entities = [
            "PERSON",           # Names of students, faculty, family members, etc.
            "STUDENT_ID",       # Student and campus IDs
            "ID_NUMBER",        # General ID numbers
            "PHONE_NUMBER",     # Phone numbers (US and international)
            "EMAIL_ADDRESS",    # Email addresses
            "US_SSN",           # Social Security Numbers
            "LOCATION",         # Street addresses, cities, locations
        ]

        # Replacement label templates
        self.default_operators = {
            "PERSON": OperatorConfig("replace", {"new_value": "[NAME_REDACTED]"}),
            "STUDENT_ID": OperatorConfig("replace", {"new_value": "[ID_REDACTED]"}),
            "ID_NUMBER": OperatorConfig("replace", {"new_value": "[ID_REDACTED]"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE_REDACTED]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL_REDACTED]"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "[SSN_REDACTED]"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "[LOCATION_REDACTED]"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "[PII_REDACTED]"})
        }
        # Known technical/conversational terms that SpaCy frequently mislabels
        self.non_pii_exclusions = {
            "ai", "claude", "gemini", "quad", "linkedin", "slack", "zoom", "copilot",
            "cornell", "tas", "cs", "info", "info 1300", "intro", "hmm", "would be"
        }
        self.timestamp_pattern = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")

    def redact_text(self, text: str, entities: Optional[List[str]] = None) -> Tuple[str, List[RecognizerResult]]:
        """
        Analyzes and redacts personal info from a text string.
        Returns: (redacted_text, list_of_detections)
        """
        if entities is None:
            entities = self.default_entities

        # 1. Analyze text using local model
        results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language="en"
        )

        # 2. Filter false positives and refine entity types
        filtered_results = []
        for r in results:
            span = text[r.start:r.end].strip()
            span_lower = span.lower()
            if self.timestamp_pattern.match(span):
                continue
            if span_lower in self.non_pii_exclusions:
                continue
            # Reclassify NetIDs that may have been tagged as LOCATION or PERSON by SpaCy
            if re.match(r"^[A-Za-z]{2,4}\d{1,5}$", span):
                r.entity_type = "STUDENT_ID"
            filtered_results.append(r)

        # 3. Anonymize/Replace with tags
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=filtered_results,
            operators=self.default_operators
        )

        # Clean up nested bracket artifacts from speaker tags, e.g. [[NAME_REDACTED]] -> [NAME_REDACTED]
        cleaned_text = re.sub(r'\[\s*\[([A-Z_]+)\]\s*\]', r'[\1]', anonymized.text)

        return cleaned_text, filtered_results

    def redact_lines(self, lines: List[str]) -> List[str]:
        """
        Redacts a list of transcript lines.
        """
        full_text = "\n".join(lines)
        redacted_full, _ = self.redact_text(full_text)
        return redacted_full.split("\n")

    def redact_vtt(self, vtt_path: str, output_path: Optional[str] = None) -> Tuple[str, List[RecognizerResult]]:
        """
        Reads a WebVTT (.vtt) transcript file, redacts PII while preserving
        timestamps and structure, and optionally writes to output_path.
        """
        if not os.path.exists(vtt_path):
            raise FileNotFoundError(f"VTT file not found: {vtt_path}")

        with open(vtt_path, "r", encoding="utf-8") as f:
            vtt_content = f.read()

        redacted_text, detections = self.redact_text(vtt_content)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(redacted_text)

        return redacted_text, detections

    def redact_pdf(self, pdf_path: str, output_txt_path: Optional[str] = None) -> Tuple[str, List[RecognizerResult]]:
        """
        Reads a PDF file, extracts text across all pages, and redacts PII.
        Optionally writes the redacted transcript to an output file.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        reader = pypdf.PdfReader(pdf_path)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        full_pdf_text = "\n\n--- PAGE BREAK ---\n\n".join(extracted_pages)
        redacted_text, detections = self.redact_text(full_pdf_text)

        if output_txt_path:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(redacted_text)

        return redacted_text, detections


# =====================================================================
# 3. CLI & TESTING INTERFACE
# =====================================================================

def test_transcript_file(redactor: LocalRedactor, target_path: str = "0926-0913-cw.vtt"):
    """
    Tests and validates redaction on the user's transcript file (0926-0913-cw.vtt).
    Verifies that all PII (names, NetID) are redacted, while domain terms (AI, Claude, etc.) are preserved.
    """
    if not os.path.exists(target_path):
        print(f"[-] Target transcript file '{target_path}' not found.")
        return

    print("=" * 75)
    print(f"[*] TESTING TRANSCRIPT REDACTION ON: {target_path}")
    print("=" * 75)

    output_path = os.path.splitext(target_path)[0] + "_redacted.vtt"
    redacted_content, detections = redactor.redact_vtt(target_path, output_path=output_path)

    # 1. Detection summary
    from collections import Counter
    entity_counts = Counter(d.entity_type for d in detections)
    print("\n[+] DETECTED PII SUMMARY:")
    for entity, count in entity_counts.items():
        print(f"    • {entity:<15}: {count} occurrences")

    # 2. Key entities audit
    with open(target_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    unique_redacted_terms = set(original_content[d.start:d.end] for d in detections)
    print(f"\n[+] UNIQUE REDACTED IDENTIFIERS: {sorted(list(unique_redacted_terms))}")

    # 3. Excerpt Comparison (Showing Introduction & NetID)
    orig_lines = original_content.splitlines()
    redact_lines = redacted_content.splitlines()

    print("\n" + "=" * 75)
    print("[*] BEFORE & AFTER EXCERPT (Lines 43 - 54: Speaker Intro & NetID):")
    print("=" * 75)
    print("\n--- ORIGINAL EXCERPT ---")
    for line in orig_lines[42:54]:
        print(f"  {line}")

    print("\n--- REDACTED EXCERPT ---")
    for line in redact_lines[42:54]:
        print(f"  {line}")

    # 4. Correctness Validations
    print("\n" + "=" * 75)
    print("[*] ACCURACY & PRESERVATION VERIFICATION:")
    print("=" * 75)
    
    checks = [
        ("Student NetID (ACH282) redacted", "ACH282" not in redacted_content),
        ("Interviewer full name (Claire Wu) redacted", "Claire Wu" not in redacted_content),
        ("Interviewee first name (Arianna) redacted", "Arianna" not in redacted_content),
        ("Technology name 'gen AI' preserved", "gen AI" in redacted_content),
        ("AI tool name 'Claude' preserved", "Claude" in redacted_content),
        ("AI tool name 'Gemini' preserved", "Gemini" in redacted_content),
    ]

    all_passed = True
    for label, passed in checks:
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {label}")
        if not passed:
            all_passed = False

    print("=" * 75)
    if all_passed:
        print(f"[✓] All checks passed successfully!")
    print(f"[✓] Redacted transcript saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local, Offline Privacy-Preserving Transcript Redactor"
    )
    parser.add_argument("--vtt", type=str, help="Path to a WebVTT (.vtt) transcript file to redact")
    parser.add_argument("--pdf", type=str, help="Path to a PDF transcript file to redact")
    parser.add_argument("--txt", type=str, help="Path to a TXT transcript file to redact")
    parser.add_argument("--text", type=str, help="Direct text string to redact")
    parser.add_argument("--output", type=str, help="Optional output path to save redacted text")
    parser.add_argument("--audit", action="store_true", help="Print audit log of detected entities")

    args = parser.parse_args()

    print("[*] Initializing local offline NLP engine (Presidio + SpaCy)...")
    redactor = LocalRedactor()
    print("[*] Local engine loaded. Ready.\n")

    # If no arguments provided, test the target transcript file directly
    if not (args.vtt or args.pdf or args.txt or args.text):
        target_file = "0926-0913-cw.vtt" if os.path.exists("0926-0913-cw.vtt") else None
        if target_file:
            test_transcript_file(redactor, target_file)
        else:
            print("[-] No input file provided. Use --vtt, --pdf, --txt, or --text.")
        return

    # VTT file input
    if args.vtt:
        output_path = args.output or (os.path.splitext(args.vtt)[0] + "_redacted.vtt")
        redacted, detections = redactor.redact_vtt(args.vtt, output_path=output_path)
        print(f"[✓] Processed VTT transcript: {args.vtt}")
        print(f"[✓] Saved redacted output to: {output_path}")
        if args.audit:
            print("\n--- AUDIT LOG ---")
            for d in detections:
                print(f"  [{d.entity_type}] (Score: {d.score:.2f})")

    # Direct text input
    elif args.text:
        redacted, detections = redactor.redact_text(args.text)
        print("--- REDACTED OUTPUT ---")
        print(redacted)
        if args.audit:
            print("\n--- AUDIT LOG ---")
            for d in detections:
                print(f"  [{d.entity_type}] (Score: {d.score:.2f})")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(redacted)
            print(f"\nSaved to: {args.output}")

    # Plain text file input
    elif args.txt:
        with open(args.txt, "r", encoding="utf-8") as f:
            content = f.read()
        redacted, detections = redactor.redact_text(content)
        print("--- REDACTED OUTPUT ---")
        print(redacted)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(redacted)
            print(f"\nSaved to: {args.output}")

    # PDF file input
    elif args.pdf:
        output_path = args.output or (os.path.splitext(args.pdf)[0] + "_redacted.txt")
        redacted, detections = redactor.redact_pdf(args.pdf, output_txt_path=output_path)
        print("--- REDACTED PDF OUTPUT ---")
        print(redacted)
        print(f"\nSaved redacted transcript to: {output_path}")
        if args.audit:
            print("\n--- AUDIT LOG ---")
            for d in detections:
                print(f"  [{d.entity_type}] (Score: {d.score:.2f})")


if __name__ == "__main__":
    main()
