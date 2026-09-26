"""LLM DNA (Wu et al., ICLR 2026, arXiv 2509.24496), rebuilt on our answer store."""

from bullpen.competitors.llmdna.encoder import LLMDNA_DIM, LLMDNA_PROMPTS, LLMDNAEncoder

__all__ = ["LLMDNA_DIM", "LLMDNA_PROMPTS", "LLMDNAEncoder"]
