"""ROI (Region of Interest) extraction module for JPA score sheets.

This module provides functions to extract specific regions from score sheet images,
such as player numbers, skill levels, scores, etc.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ROI:
    """Container for Region of Interest."""

    x: int
    y: int
    width: int
    height: int
    label: str
    image: Optional[np.ndarray] = None

    def extract(self, source_image: np.ndarray) -> np.ndarray:
        """Extract this ROI from source image.

        Args:
            source_image: Source image to extract from

        Returns:
            Extracted ROI image
        """
        roi_image = source_image[self.y:self.y+self.height, self.x:self.x+self.width]
        self.image = roi_image
        return roi_image

    def to_dict(self) -> Dict:
        """Convert ROI to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'label': self.label
        }


class ROIExtractor:
    """Handles ROI extraction for JPA score sheets."""

    def __init__(self):
        """Initialize ROIExtractor."""
        pass

    def separate_sheets(
        self,
        image: np.ndarray,
        visualize: bool = False
    ) -> List[np.ndarray]:
        """Separate 2 sheets from a single photo.

        Args:
            image: Input image containing 2 sheets
            visualize: Whether to save visualization

        Returns:
            List of 2 separated sheet images
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply threshold to find paper edges
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (paper should be white)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area (find the 2 largest rectangles)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        sheets = []
        for i, contour in enumerate(contours[:2]):  # Top 2 largest contours
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out noise (must be reasonably large)
            if w > image.shape[1] * 0.3 and h > image.shape[0] * 0.3:
                sheet = image[y:y+h, x:x+w]
                sheets.append(sheet)

        # Sort sheets by y-position (top to bottom)
        if len(sheets) == 2:
            sheets = sorted(sheets, key=lambda s: cv2.boundingRect(
                cv2.findContours(
                    cv2.threshold(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )[0][0]
            )[1])

        return sheets

    def detect_match_areas(
        self,
        sheet: np.ndarray,
        num_matches: int = 5
    ) -> List[ROI]:
        """Detect individual match record areas in a sheet.

        Args:
            sheet: Single sheet image (1 team)
            num_matches: Expected number of matches (default: 5)

        Returns:
            List of ROI objects for each match area
        """
        # For now, divide sheet vertically into equal parts
        # TODO: Use line detection for more accurate segmentation

        height, width = sheet.shape[:2]
        match_height = height // num_matches

        match_rois = []
        for i in range(num_matches):
            y = i * match_height
            roi = ROI(
                x=0,
                y=y,
                width=width,
                height=match_height,
                label=f"match_{i+1}"
            )
            match_rois.append(roi)

        return match_rois

    def extract_player_number_roi(
        self,
        match_area: np.ndarray
    ) -> ROI:
        """Extract player number ROI from match area.

        Args:
            match_area: Image of single match record area

        Returns:
            ROI for player number
        """
        # Player number is at the left edge
        # Adjusted to avoid left border line
        height, width = match_area.shape[:2]

        roi = ROI(
            x=int(width * 0.025),  # 2.5% from left (avoid border)
            y=int(height * 0.25),  # 25% from top (more centered)
            width=int(width * 0.055),  # 5.5% of width (narrower, just the number)
            height=int(height * 0.35),  # 35% of height
            label="player_number"
        )

        return roi

    def extract_skill_level_roi(
        self,
        match_area: np.ndarray
    ) -> ROI:
        """Extract skill level ROI from match area.

        Skill level is in a small diagonal box to the right of player name.

        Args:
            match_area: Image of single match record area

        Returns:
            ROI for skill level
        """
        # Skill level is in diagonal box, right side of name area
        # Need to capture just the upper-left triangle of the diagonal box
        height, width = match_area.shape[:2]

        roi = ROI(
            x=int(width * 0.235),  # 23.5% from left (more precise)
            y=int(height * 0.20),  # 20% from top (lower to avoid top border)
            width=int(width * 0.04),  # 4% of width (smaller, just the digit)
            height=int(height * 0.25),  # 25% of height (smaller box)
            label="skill_level"
        )

        return roi

    def extract_total_score_roi(
        self,
        match_area: np.ndarray
    ) -> ROI:
        """Extract total score ROI from match area.

        Total score is typically on the right side of the sheet.

        Args:
            match_area: Image of single match record area

        Returns:
            ROI for total score
        """
        height, width = match_area.shape[:2]

        roi = ROI(
            x=int(width * 0.855),  # 85.5% from left (more right, specific cell)
            y=int(height * 0.25),  # 25% from top (more centered)
            width=int(width * 0.065),  # 6.5% of width (narrower, single cell)
            height=int(height * 0.30),  # 30% of height (smaller box)
            label="total_score"
        )

        return roi

    def extract_all_fields(
        self,
        match_area: np.ndarray
    ) -> Dict[str, ROI]:
        """Extract all field ROIs from a match area.

        Args:
            match_area: Image of single match record area

        Returns:
            Dictionary of field name to ROI
        """
        fields = {}

        # Extract each field
        fields['player_number'] = self.extract_player_number_roi(match_area)
        fields['skill_level'] = self.extract_skill_level_roi(match_area)
        fields['total_score'] = self.extract_total_score_roi(match_area)

        # Extract ROI images
        for field_name, roi in fields.items():
            roi.extract(match_area)

        return fields

    def process_full_scoresheet(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """Process complete scoresheet image and extract all ROIs.

        Args:
            image: Full scoresheet image (containing 2 sheets)

        Returns:
            List of dictionaries containing extracted data per match
        """
        results = []

        # Step 1: Separate sheets
        sheets = self.separate_sheets(image)

        if len(sheets) != 2:
            print(f"Warning: Expected 2 sheets, found {len(sheets)}")

        # Process each sheet
        for sheet_idx, sheet in enumerate(sheets):
            # Step 2: Detect match areas
            match_areas = self.detect_match_areas(sheet)

            # Step 3: Extract fields from each match
            for match_idx, match_roi in enumerate(match_areas):
                match_area = match_roi.extract(sheet)

                # Extract all fields
                fields = self.extract_all_fields(match_area)

                result = {
                    'sheet_index': sheet_idx,
                    'match_index': match_idx,
                    'match_label': match_roi.label,
                    'fields': fields
                }

                results.append(result)

        return results


def visualize_rois(
    image: np.ndarray,
    rois: List[ROI],
    output_path: str
) -> None:
    """Visualize ROIs on image and save.

    Args:
        image: Source image
        rois: List of ROIs to visualize
        output_path: Path to save visualization
    """
    vis_image = image.copy()

    for roi in rois:
        # Draw rectangle
        cv2.rectangle(
            vis_image,
            (roi.x, roi.y),
            (roi.x + roi.width, roi.y + roi.height),
            (0, 255, 0),
            2
        )

        # Add label
        cv2.putText(
            vis_image,
            roi.label,
            (roi.x, roi.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, vis_image)
