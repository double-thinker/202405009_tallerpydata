def ascii_to_tag_space(text):
    tag_space_chars = []
    for char in text:
        if ord(char) >= 32 and ord(char) <= 126:
            tag_space_char = chr(ord(char) + 917504)
            tag_space_chars.append(tag_space_char)
        else:
            tag_space_chars.append(char)
    return "".join(tag_space_chars)
