


def read_markdown_file(file_path: str) -> str:
    """
    Lê um arquivo Markdown e retorna seu conteúdo como uma string.
        
    :param file_path: Caminho para o arquivo Markdown.
    :return: Conteúdo do arquivo como string.
        """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content