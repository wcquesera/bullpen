"""LOCUS (Patel, Cocke & Joshi, arXiv 2601.21082), reimplemented."""

from bullpen.competitors.locus.encoder import LOCUS_DIM, LocusEncoder
from bullpen.competitors.locus.net import LocusConfig, LocusNet

__all__ = ["LOCUS_DIM", "LocusConfig", "LocusEncoder", "LocusNet"]
