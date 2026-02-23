import re


def split_text(text,
               max_chars=700,
               overlap=120):

    text = re.sub(r"\s+", " ", text)

    paragraphs = re.split(r"\n{2,}", text)

    chunks = []
    current = ""

    for para in paragraphs:

        para = para.strip()

        if len(para) < 40:
            continue

        sentences = re.split(r"(?<=[।.!?])", para)

        for sent in sentences:

            sent = sent.strip()

            if not sent:
                continue

            if len(current) + len(sent) > max_chars:

                chunks.append(current.strip())

                current = current[-overlap:] + " " + sent

            else:
                current += " " + sent

    if current.strip():
        chunks.append(current.strip())

    chunks = [c for c in chunks if len(c) > 80]

    return chunks