from pathlib import Path
from grid_extraction.oop import GridExtractor

def extract_instructions_from_image(
    img_path: Path,
    start_corner: str,
    crochet: bool,
) -> list[str]:
    extractor = GridExtractor(str(img_path))

    extractor.load_image()
    extractor.preprocess()
    extractor.detect_lines()
    extractor.extract_grid_lines()
    extractor.extract_cells()

    # NOTE: I’m mapping crochet -> alternate direction per row
    instructions = extractor.build_instructions(start_corner=start_corner, alternate=crochet)
    return instructions
