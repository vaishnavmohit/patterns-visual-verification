"""Pydantic schemas for structured LLM output."""

from typing import Literal

from pydantic import BaseModel, Field


class ImageDescriptionSynthLogo(BaseModel):
    """Schema for Bongard-LOGO classification responses."""

    Analysis: str
    Rule: str
    TestImage: str = Field(alias="Test Image")
    Conclusion: Literal["pos", "neg"]

    class Config:
        populate_by_name = True
