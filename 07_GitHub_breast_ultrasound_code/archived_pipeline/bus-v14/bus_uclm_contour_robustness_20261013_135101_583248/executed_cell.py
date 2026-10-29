# 修复模型位于 models/ 子目录时的读取问题
from pathlib import Path
import zipfile

def read_bytes(source, name):
    source = Path(source)
    basename = Path(name).name

    if source.is_dir():
        exact = source / name
        if exact.is_file():
            return exact.read_bytes()

        matches = [
            p for p in source.rglob(basename)
            if p.is_file() and "__MACOSX" not in p.parts
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{source} 内找到 {len(matches)} 个 {basename}：\n"
                + "\n".join(str(p) for p in matches)
            )
        return matches[0].read_bytes()

    with zipfile.ZipFile(source) as z:
        matches = [
            n for n in z.namelist()
            if not n.endswith("/")
            and Path(n).name == basename
            and "__MACOSX" not in n
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{source.name} 内找到 {len(matches)} 个 {basename}"
            )
        return z.read(matches[0])

# 使用修复后的函数继续运行，原模型哈希检查仍保留。
main()