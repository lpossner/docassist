import fitz


def chunk_pdf_(path):
    doc = fitz.open(path)
    # Get bookmarks
    bookmarks = doc.get_toc()
    # Extract pages
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))

    # Extract full chapter titles
    chapter_titles = []
    page_numbers = []
    # Get deepest chapter level
    max_level = max([level for level, _, _ in bookmarks])
    current_title = [''] * max_level
    old_level = 0
    for level, titel, page_number in bookmarks:
        # Add title of current page chunks
        current_title[level - 1] = titel
        # If level decreases, roll back
        if old_level > level:
            # Remove titles up to current level
            for remove_index in range(level, old_level):
                current_title[remove_index] = ''
        chapter_titles.append(current_title[:])
        page_numbers.append(page_number)
        old_level = level

    # Extract pages of a chapter and their numbers
    chapter_pages = []
    chapter_page_numbers = []
    for page_number_start, page_number_stop in zip(page_numbers, page_numbers[1:]):
        chapter_pages.append(texts[page_number_start - 1: page_number_stop - 1])
        chapter_page_numbers.append(list(range(page_number_start, page_number_stop)))
    chapter_pages.append(chapter_page_numbers[page_number_start: page_number_stop])
    chapter_page_numbers.append(list(range(page_number_start + 1, page_number_stop + 1)))

    return chapter_pages, chapter_page_numbers, chapter_titles
