"""
CoMpaNeoN LearnWords
====================

Lexical learning-unit construction layer.

Pipeline:

    DataFilter
        ↓
    LearnWords
        ↓
    WordChain
        ↓
    WordUnderstanding

LearnWords is responsible for:

- language detection
- tokenizer integration
- canonical lexical extraction
- irrelevant abbreviation filtering
- preservation of original lexical forms
- multilingual structural representation
- words
- phrases
- sentences
- paragraphs
- linguistic sequence construction
- WordChain ingestion

LearnWords does NOT:

- replace tokenizer.py
- change tokenizer mathematics
- perform GSP traversal
- perform MemoryGrid placement
- perform MemoryPartition routing
- perform final model training
- replace WordUnderstanding
- replace FineTunerAndWeightScalar
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

from langdetect import detect, LangDetectException

try:
    from .tokenizer import (
        tokenize,
        normalize_lang,
    )
    from .word_chain import WordChain
except ImportError:
    from tokenizer import (
        tokenize,
        normalize_lang,
    )
    from word_chain import WordChain


# ============================================================================
# CANONICAL ABBREVIATIONS
# ============================================================================

# These are retained because they are meaningful lexical/technical units.
# This list is deliberately conservative.

CANONICAL_ABBREVIATIONS = {
    # computing
    "cpu",
    "gpu",
    "ram",
    "rom",
    "arm",
    "x86",
    "x64",
    "nvme",
    "ssd",
    "hdd",
    "ecc",
    "ddr",
    "ddr4",
    "ddr5",
    "hbm",
    "hbm2",
    "hbm3",
    "hbm3e",
    "usb",
    "pci",
    "pcie",
    "nvlink",

    # software
    "api",
    "sdk",
    "orm",
    "sql",
    "http",
    "https",
    "html",
    "css",
    "json",
    "xml",
    "yaml",
    "rest",
    "jwt",
    "otp",
    "url",
    "uri",
    "uid",
    "uuid",
    "ip",
    "dns",
    "tcp",
    "udp",
    "ssh",
    "ssl",
    "tls",

    # operating systems / infrastructure
    "os",
    "vm",
    "kvm",
    "vps",
    "cdn",
    "ram",
    "lan",
    "wan",

    # AI
    "ai",
    "ml",
    "llm",
    "nlp",
    "cuda",
    "cudnn",
    "opencl",

    # common academic/technical
    "phd",
    "msc",
    "bsc",
    "lga",
    "ltd",
    "inc",
}


# ============================================================================
# IRRELEVANT ABBREVIATIONS
# ============================================================================

# These should not become canonical learning words unless explicitly
# overridden by metadata.

IRRELEVANT_ABBREVIATIONS = {
    "etc",
    "e.g",
    "eg",
    "i.e",
    "ie",
    "aka",
    "approx",
    "misc",
    "dept",
    "est",
    "fig",
    "no",
    "nos",
    "vol",
    "vs",
    "viz",
    "msg",
    "info",
    "ref",
    "temp",
    "misc",
}


# ============================================================================
# CLASSIC SAMPLE CORPUS
# ============================================================================

CLASSIC_LEARNING_BLOCKS = {

    "daily_conversation": {

        "words": [
            "hello",
            "goodbye",
            "please",
            "thanks",
            "sorry",
            "yes",
            "no",
            "maybe",
            "friend",
            "family",
            "home",
            "work",
            "school",
            "today",
            "tomorrow",
            "help",
            "understand",
            "explain",
            "remember",
            "forget",
        ],

        "phrases": [
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "I am fine",
            "thank you",
            "you are welcome",
            "please help me",
            "I understand",
            "I do not understand",
            "can you explain",
            "what do you mean",
            "let me check",
            "give me a moment",
            "that is correct",
            "that is not correct",
        ],

        "sentences": [
            "Hello, how are you today?",
            "I am fine, thank you.",
            "Please help me understand this.",
            "Can you explain what you mean?",
            "I understand the idea.",
            "I do not understand the last part.",
            "Let me check the information.",
            "That is correct.",
            "That is not correct.",
            "Please give me a moment.",
            "We can continue the discussion tomorrow.",
            "I will explain the problem clearly.",
        ],

        "paragraphs": [
            (
                "A conversation begins with a clear exchange between "
                "people. One person may ask a question, another person "
                "may answer, and either person may ask for clarification. "
                "Good communication depends on understanding what was "
                "said before responding."
            ),
            (
                "When a person does not understand something, the person "
                "can ask for an explanation. The explanation may be short "
                "or detailed depending on the subject. A useful response "
                "should address the actual question instead of assuming "
                "what the person meant."
            ),
        ],
    },

    "questions_and_explanations": {

        "words": [
            "what",
            "why",
            "when",
            "where",
            "who",
            "which",
            "how",
            "difference",
            "reason",
            "example",
            "meaning",
            "definition",
            "explanation",
        ],

        "phrases": [
            "what is",
            "what does it mean",
            "why does this happen",
            "how does it work",
            "what is the difference",
            "give an example",
            "explain the reason",
            "define the term",
        ],

        "sentences": [
            "What is this system?",
            "What does this word mean?",
            "Why does this happen?",
            "How does the system work?",
            "What is the difference between the two systems?",
            "Can you give an example?",
            "Please explain the reason.",
            "What does this result mean?",
            "How can the problem be solved?",
        ],

        "paragraphs": [
            (
                "A question identifies something that the speaker wants "
                "to understand. The answer should identify the subject, "
                "explain the relevant relationship, and provide an example "
                "when an example makes the idea easier to understand."
            ),
        ],
    },

    "mathematics": {

        "words": [
            "number",
            "value",
            "sum",
            "difference",
            "product",
            "ratio",
            "equation",
            "variable",
            "matrix",
            "vector",
            "coordinate",
            "function",
            "set",
            "factor",
            "constant",
        ],

        "phrases": [
            "add two numbers",
            "calculate the sum",
            "find the difference",
            "multiply the values",
            "solve the equation",
            "calculate the ratio",
            "find the coordinate",
            "compare two values",
            "calculate the vector",
            "evaluate the function",
        ],

        "sentences": [
            "The sum of two numbers is calculated by adding their values.",
            "The difference between two numbers is found by subtraction.",
            "A variable represents a value that may change.",
            "A coordinate identifies a position in a grid.",
            "A vector can represent direction and magnitude.",
            "A matrix contains values arranged in rows and columns.",
            "The same input should produce the same deterministic result.",
        ],

        "paragraphs": [
            (
                "A mathematical system uses defined operations to transform "
                "values. An equation may contain constants and variables, "
                "while a matrix may contain values arranged into rows and "
                "columns. A deterministic calculation produces the same "
                "result whenever the same valid inputs are supplied."
            ),
        ],
    },

    "engineering": {

        "words": [
            "system",
            "server",
            "processor",
            "memory",
            "storage",
            "network",
            "database",
            "kernel",
            "container",
            "device",
            "sensor",
            "controller",
            "architecture",
            "component",
            "interface",
        ],

        "phrases": [
            "computer system",
            "server processor",
            "memory controller",
            "storage device",
            "database server",
            "network interface",
            "system architecture",
            "software component",
            "hardware component",
            "distributed system",
        ],

        "sentences": [
            "A computer system contains hardware and software components.",
            "A server provides resources or services to other systems.",
            "A database stores structured information for later retrieval.",
            "A network interface connects a device to a network.",
            "A software component performs a defined responsibility.",
            "System architecture describes how components work together.",
        ],

        "paragraphs": [
            (
                "A computing system may contain processors, memory, storage, "
                "network interfaces, operating-system services, and application "
                "software. Each component has a defined responsibility. A good "
                "architecture keeps those responsibilities separated while "
                "allowing the components to communicate through clear interfaces."
            ),
        ],
    },

    "coding": {

        "words": [
            "code",
            "function",
            "class",
            "object",
            "variable",
            "method",
            "module",
            "package",
            "route",
            "request",
            "response",
            "database",
            "query",
            "error",
            "debug",
        ],

        "phrases": [
            "write a function",
            "call a function",
            "create an object",
            "define a class",
            "import a module",
            "send a request",
            "return a response",
            "query the database",
            "handle an error",
            "debug the program",
        ],

        "sentences": [
            "A function performs a defined operation.",
            "A class defines the structure and behavior of objects.",
            "A module groups related functionality.",
            "A request can be sent to a server endpoint.",
            "A response contains the result returned by the server.",
            "An error should be identified before the system is changed.",
            "A database query retrieves information from stored data.",
        ],

        "paragraphs": [
            (
                "A software application is usually divided into components "
                "with specific responsibilities. A function may perform a "
                "calculation, a class may represent an object, and a module "
                "may group related functionality. Clear separation makes "
                "the system easier to understand, test, and extend."
            ),
        ],
    },

    "education": {

        "words": [
            "student",
            "teacher",
            "lesson",
            "class",
            "subject",
            "question",
            "answer",
            "knowledge",
            "practice",
            "example",
            "exercise",
            "understanding",
        ],

        "phrases": [
            "learn a concept",
            "study a subject",
            "answer a question",
            "solve an exercise",
            "give an example",
            "practice a skill",
            "understand the lesson",
        ],

        "sentences": [
            "A student learns by studying information and practicing skills.",
            "A teacher can explain a difficult concept with an example.",
            "An exercise allows a student to practice a concept.",
            "A question can reveal what a student understands.",
            "Practice helps reinforce previously learned information.",
        ],

        "paragraphs": [
            (
                "Learning develops through repeated exposure, understanding, "
                "practice, comparison, and correction. A learner may first "
                "encounter a word, then see the word in a phrase, sentence, "
                "and paragraph. The surrounding context helps the learner "
                "understand how the word behaves in real communication."
            ),
        ],
    },

    "business": {

        "words": [
            "business",
            "customer",
            "merchant",
            "product",
            "service",
            "price",
            "payment",
            "order",
            "receipt",
            "invoice",
            "account",
            "transaction",
        ],

        "phrases": [
            "buy a product",
            "sell a product",
            "make a payment",
            "place an order",
            "receive a receipt",
            "create an invoice",
            "complete a transaction",
            "customer account",
            "merchant account",
        ],

        "sentences": [
            "A customer can place an order for a product.",
            "A merchant provides products or services to customers.",
            "A payment records the transfer of value for a transaction.",
            "A receipt records a completed purchase.",
            "An invoice describes an amount that is due.",
        ],

        "paragraphs": [
            (
                "A transaction connects a customer, a merchant, a product "
                "or service, and a payment process. An order describes what "
                "the customer wants to obtain. A receipt can confirm a "
                "completed purchase, while an invoice can describe an amount "
                "that remains due."
            ),
        ],
    },

    "organization_ai": {

        "words": [
            "organization",
            "worker",
            "member",
            "manager",
            "team",
            "department",
            "project",
            "role",
            "permission",
            "privacy",
            "memory",
            "context",
        ],

        "phrases": [
            "organization memory",
            "personal memory",
            "project context",
            "worker identity",
            "team member",
            "department role",
            "access permission",
            "private information",
            "shared knowledge",
        ],

        "sentences": [
            "An organization contains people with different roles.",
            "A worker may belong to a team or department.",
            "Project context describes the work currently being performed.",
            "Private information should not automatically become shared knowledge.",
            "Permissions determine which information a member can access.",
            "Personal memory and organization memory can have different boundaries.",
        ],

        "paragraphs": [
            (
                "An organization-aware AI must distinguish between personal "
                "context and shared organizational knowledge. A worker may "
                "have private information that should remain private, while "
                "the organization may maintain knowledge that is available "
                "to authorized members. Access should therefore follow the "
                "defined role, permission, and context boundaries."
            ),
        ],
    },

    "technical_infrastructure": {

        "words": [
            "CPU",
            "ARM",
            "x86",
            "ECC",
            "DDR5",
            "NVMe",
            "Linux",
            "kernel",
            "KVM",
            "container",
            "GPU",
            "Tensor",
            "Core",
            "VRAM",
            "HBM",
            "CUDA",
            "cuDNN",
            "OpenCL",
            "Nginx",
            "Apache",
            "OpenSSL",
            "cache",
        ],

        "phrases": [
            "server CPU",
            "ECC memory",
            "DDR5 memory",
            "NVMe storage",
            "Linux kernel",
            "KVM virtual machine",
            "GPU memory",
            "HBM memory",
            "CUDA application",
            "OpenCL application",
            "web server",
            "application server",
            "AI training server",
        ],

        "sentences": [
            "A server CPU executes instructions for the operating system and applications.",
            "ECC memory can detect and correct certain memory errors.",
            "NVMe storage provides a high-speed interface for storage devices.",
            "The Linux kernel manages hardware and system resources.",
            "A container isolates an application environment from other processes.",
            "A GPU can accelerate parallel workloads.",
            "HBM provides high-bandwidth memory access for supported processors.",
            "A web server receives requests and returns responses.",
            "An AI training server can contain multiple GPUs connected by a high-speed interconnect.",
        ],

        "paragraphs": [
            (
                "A modern computing infrastructure can contain server CPUs, "
                "ECC DDR5 memory, NVMe storage, Linux, virtualization, "
                "containers, GPUs, high-bandwidth memory, and networking. "
                "Each layer provides a different responsibility. The "
                "application depends on the operating system and hardware "
                "while the infrastructure provides the resources required "
                "to execute the workload."
            ),
        ],
    },
}


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect language with langdetect, then normalize through tokenizer.py.
    """

    if not text or not str(text).strip():
        return "en"

    try:
        detected = detect(
            str(text)
        )

        return normalize_lang(
            detected
        )

    except LangDetectException:
        return "en"

    except Exception:
        return "en"


# ============================================================================
# TEXT / ABBREVIATION HELPERS
# ============================================================================

_WORDLIKE_RE = re.compile(
    r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$",
    flags=re.UNICODE,
)

_LATIN_ABBREVIATION_RE = re.compile(
    r"^[A-Z]{1,8}$"
)

_DOTTED_ABBREVIATION_RE = re.compile(
    r"^(?:[A-Za-z]\.){2,}$"
)


def _surface_word(
    token: Mapping[str, Any],
) -> str:

    return str(
        token.