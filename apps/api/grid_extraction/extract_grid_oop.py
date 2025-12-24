from oop import GridExtractor

extractor = GridExtractor("C:/Users/nikol/Downloads/page_5.png")

extractor.load_image()
extractor.preprocess()
extractor.detect_lines()
extractor.extract_grid_lines()
extractor.extract_cells()

instructions = extractor.build_instructions(start_corner="bottom-left", alternate=False)
# print(extractor.pattern_labels)
# print("\n\n\n")
# print(instructions)
extractor.show_results()

for i in instructions:
    print(i)
