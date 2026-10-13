#!/usr/bin/env python3
import argparse
import ast
import io
import json
import os
import re
import shutil
import tokenize
from pathlib import Path

EXTENSIONES_POR_DEFECTO = ".py,.pyw,.pyi,.ipynb"

CARPETAS_EXCLUIDAS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    "site-packages", ".tox", ".nox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".ipynb_checkpoints",
}

CODING_RE = re.compile(r"^#.*?coding[:=][ \t]*[-\w.]+")
MAGIA_RE = re.compile(r"^([ \t]*)[%!]")
MAGIAS_DE_CELDA_PYTHON = {"time", "timeit", "capture", "prun"}


def partir_lineas(texto):
    return io.StringIO(texto, newline="").readlines()


def fin_de_linea(linea):
    return linea[len(linea.rstrip("\r\n")):]


def vistas(lineas, notebook):
    yield lineas
    if notebook:
        enmascaradas = []
        for linea in lineas:
            m = MAGIA_RE.match(linea)
            enmascaradas.append(m.group(1) + "pass" + fin_de_linea(linea) if m else linea)
        yield enmascaradas


def analizar(lineas, notebook):
    for vista in vistas(lineas, notebook):
        try:
            return ast.parse("".join(vista)), vista
        except (SyntaxError, ValueError):
            pass
    return None, None


def localizar_comentarios(lineas, notebook):
    error = None
    for vista in vistas(lineas, notebook):
        siguiente = iter(vista)
        try:
            return [
                (tok.start[0], tok.start[1], tok.string)
                for tok in tokenize.generate_tokens(lambda: next(siguiente, ""))
                if tok.type == tokenize.COMMENT
            ]
        except (tokenize.TokenError, SyntaxError) as e:
            error = e
    raise ValueError(f"no se puede tokenizar ({error})")


def quitar_comentarios(lineas, notebook, conservar):
    nuevas = list(lineas)
    borrar = set()
    quitados = 0
    for fila, col, texto in localizar_comentarios(lineas, notebook):
        if conservar(fila, texto):
            continue
        i = fila - 1
        codigo = nuevas[i][:col].rstrip()
        if codigo:
            nuevas[i] = codigo + fin_de_linea(nuevas[i])
        else:
            borrar.add(i)
        quitados += 1
    return [linea for i, linea in enumerate(nuevas) if i not in borrar], quitados


def lineas_dentro_de_cadenas(lineas, notebook):
    for vista in vistas(lineas, notebook):
        siguiente = iter(vista)
        try:
            dentro = set()
            for tok in tokenize.generate_tokens(lambda: next(siguiente, "")):
                if tok.start[0] != tok.end[0]:
                    dentro.update(range(tok.start[0], tok.end[0] + 1))
            return dentro
        except (tokenize.TokenError, SyntaxError):
            pass
    return None


def ordenar_lineas_en_blanco(lineas, notebook):
    dentro = lineas_dentro_de_cadenas(lineas, notebook)
    if dentro is None:
        return lineas

    def en_blanco(i):
        if lineas[i].strip() or (i + 1) in dentro:
            return False
        anterior = lineas[i - 1].rstrip("\r\n") if i > 0 else ""
        return not anterior.endswith("\\")

    resultado = []
    seguidas = 0
    solo_cabecera = True
    for i, linea in enumerate(lineas):
        if en_blanco(i):
            seguidas += 1
            limite = 1 if solo_cabecera else 2
            if resultado and seguidas <= limite:
                resultado.append((i, linea))
        else:
            seguidas = 0
            if not linea.lstrip().startswith("#"):
                solo_cabecera = False
            resultado.append((i, linea))
    while resultado and en_blanco(resultado[-1][0]):
        resultado.pop()
    return [linea for _, linea in resultado]


def es_cadena_suelta(nodo):
    return (
        isinstance(nodo, ast.Expr)
        and isinstance(nodo.value, ast.Constant)
        and isinstance(nodo.value.value, str)
    )


def ocupa_lineas_completas(lineas, nodo):
    antes = lineas[nodo.lineno - 1].encode("utf-8")[:nodo.col_offset]
    despues = lineas[nodo.end_lineno - 1].encode("utf-8")[nodo.end_col_offset:]
    return not antes.strip() and not despues.strip()


def quitar_cadenas(lineas, notebook):
    arbol, vista = analizar(lineas, notebook)
    if arbol is None:
        return lineas, 0, False
    acciones = []
    for nodo in ast.walk(arbol):
        for campo in ("body", "orelse", "finalbody"):
            bloque = getattr(nodo, campo, None)
            if not isinstance(bloque, list) or not bloque:
                continue
            sueltas = [s for s in bloque
                       if es_cadena_suelta(s) and ocupa_lineas_completas(vista, s)]
            if not sueltas:
                continue
            queda_vacio = len(sueltas) == len(bloque) and not isinstance(nodo, ast.Module)
            for k, s in enumerate(sueltas):
                acciones.append((s.lineno, s.end_lineno, queda_vacio and k == 0))

    nuevas = list(lineas)
    for ini, fin, poner_pass in sorted(acciones, reverse=True):
        if poner_pass:
            primera = nuevas[ini - 1]
            sangria = primera[:len(primera) - len(primera.lstrip())]
            nuevas[ini - 1:fin] = [sangria + "pass" + (fin_de_linea(nuevas[fin - 1]) or "\n")]
        else:
            del nuevas[ini - 1:fin]
    return nuevas, len(acciones), True


def limpiar_codigo(texto, opciones, notebook=False):
    original = partir_lineas(texto)
    lineas, n_com = quitar_comentarios(original, notebook, opciones.conservar)
    n_cad, aviso = 0, None
    if opciones.docstrings:
        lineas, n_cad, ok = quitar_cadenas(lineas, notebook)
        if not ok:
            aviso = "no es Python 3 válido; docstrings sin tocar"
    if n_com or n_cad:
        lineas = ordenar_lineas_en_blanco(lineas, notebook)
        if notebook and lineas and not texto.endswith(("\n", "\r")):
            lineas[-1] = lineas[-1].rstrip("\r\n")

    antes, _ = analizar(original, notebook)
    despues, _ = analizar(lineas, notebook)
    if antes is not None:
        if despues is None:
            raise ValueError("el resultado no compilaría; no se modifica")
        if not opciones.docstrings and ast.dump(antes) != ast.dump(despues):
            raise ValueError("el resultado alteraría el programa; no se modifica")
    elif aviso is None:
        aviso = "no es Python 3 válido; comentarios quitados sin poder verificar"
    return "".join(lineas), n_com, n_cad, aviso


def procesar_py(ruta, opciones):
    datos = ruta.read_bytes()
    codificacion, _ = tokenize.detect_encoding(io.BytesIO(datos).readline)
    texto = datos.decode(codificacion)
    nuevo, n_com, n_cad, aviso = limpiar_codigo(texto, opciones)
    salida = nuevo.encode(codificacion) if nuevo != texto else None
    return salida, n_com, n_cad, aviso


def procesar_ipynb(ruta, opciones):
    crudo = ruta.read_text(encoding="utf-8")
    nb = json.loads(crudo)
    tot_com = tot_cad = 0
    avisos = []
    cambiado = False
    for num, celda in enumerate(nb.get("cells", []), 1):
        if celda.get("cell_type") != "code":
            continue
        fuente = celda.get("source", "")
        texto = "".join(fuente) if isinstance(fuente, list) else fuente
        inicio = texto.lstrip()
        if inicio.startswith("%%"):
            nombre = inicio[2:].split(None, 1)[0] if inicio[2:].strip() else ""
            if nombre not in MAGIAS_DE_CELDA_PYTHON:
                continue
        try:
            nuevo, n_com, n_cad, aviso = limpiar_codigo(texto, opciones, notebook=True)
        except ValueError as e:
            avisos.append(f"celda {num} omitida: {e}")
            continue
        if aviso:
            avisos.append(f"celda {num}: {aviso}")
        if nuevo != texto:
            celda["source"] = partir_lineas(nuevo) if isinstance(fuente, list) else nuevo
            cambiado = True
            tot_com += n_com
            tot_cad += n_cad
    aviso_total = "; ".join(avisos) or None
    if not cambiado:
        return None, 0, 0, aviso_total
    m = re.search(r'^( +)"', crudo, re.M)
    sangria = len(m.group(1)) if m else 1
    salida = json.dumps(nb, indent=sangria, ensure_ascii=False) + "\n"
    return salida.encode("utf-8"), tot_com, tot_cad, aviso_total


def es_entorno_virtual(carpeta):
    return (Path(carpeta) / "pyvenv.cfg").exists()


def recorrer(raiz, extensiones, excluidas):
    for carpeta, subcarpetas, archivos in os.walk(raiz):
        subcarpetas[:] = sorted(
            d for d in subcarpetas
            if d not in excluidas and not es_entorno_virtual(Path(carpeta, d))
        )
        for nombre in sorted(archivos):
            if Path(nombre).suffix.lower() in extensiones:
                yield Path(carpeta, nombre)


def main():
    p = argparse.ArgumentParser(
        description="Elimina los comentarios de los archivos Python de una carpeta y sus subcarpetas.",
        epilog="Sin -o ni --in-place solo se hace una simulación y no se modifica nada.",
    )
    p.add_argument("origen", type=Path, help="carpeta a procesar")
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("-o", "--output", type=Path, metavar="DESTINO",
                      help="crea una copia limpia de la carpeta en DESTINO (el original no se toca)")
    modo.add_argument("--in-place", action="store_true",
                      help="modifica directamente los archivos originales")
    p.add_argument("--backup", action="store_true",
                   help="con --in-place, guarda una copia .bak de cada archivo modificado")
    p.add_argument("--docstrings", action="store_true",
                   help="elimina también docstrings y bloques de cadena sueltos")
    p.add_argument("--keep", action="append", default=[], metavar="REGEX",
                   help="conserva los comentarios que encajen con REGEX (repetible)")
    p.add_argument("--ext", default=EXTENSIONES_POR_DEFECTO,
                   help=f"extensiones separadas por comas (por defecto {EXTENSIONES_POR_DEFECTO})")
    p.add_argument("--exclude", action="append", default=[], metavar="CARPETA",
                   help="nombre de carpeta a ignorar (repetible)")
    args = p.parse_args()

    origen = args.origen.resolve()
    if not origen.is_dir():
        p.error(f"{origen} no es una carpeta")
    if args.backup and not args.in_place:
        p.error("--backup solo tiene sentido junto con --in-place")

    patrones = [re.compile(r) for r in args.keep]

    def conservar(fila, texto):
        if fila == 1 and texto.startswith("#!"):
            return True
        if fila <= 2 and CODING_RE.match(texto):
            return True
        return any(pt.search(texto) for pt in patrones)

    opciones = argparse.Namespace(docstrings=args.docstrings, conservar=conservar)
    extensiones = set()
    for e in args.ext.split(","):
        e = e.strip().lower()
        if e:
            extensiones.add(e if e.startswith(".") else "." + e)
    excluidas = CARPETAS_EXCLUIDAS | set(args.exclude)

    if args.output:
        destino = args.output.resolve()
        if destino.exists():
            p.error(f"{destino} ya existe; indica una carpeta nueva")
        if origen in destino.parents:
            p.error("la carpeta de destino no puede estar dentro de la de origen")

        def ignorar(carpeta, nombres):
            return {n for n in nombres
                    if n in excluidas or n.endswith(".pyc")
                    or es_entorno_virtual(Path(carpeta, n))}

        shutil.copytree(origen, destino, symlinks=True, ignore=ignorar)
        raiz, escribir = destino, True
    elif args.in_place:
        raiz, escribir = origen, True
    else:
        raiz, escribir = origen, False
        print("SIMULACIÓN: no se modifica nada. Usa -o DESTINO o --in-place para aplicar.\n")

    n_arch = n_mod = tot_com = tot_cad = 0
    avisos = []
    for ruta in recorrer(raiz, extensiones, excluidas):
        n_arch += 1
        rel = ruta.relative_to(raiz)
        try:
            if ruta.suffix.lower() == ".ipynb":
                nuevo, n_com, n_cad, aviso = procesar_ipynb(ruta, opciones)
            else:
                nuevo, n_com, n_cad, aviso = procesar_py(ruta, opciones)
        except Exception as e:
            avisos.append(f"{rel}: omitido ({e})")
            continue
        if aviso:
            avisos.append(f"{rel}: {aviso}")
        if nuevo is None:
            continue
        n_mod += 1
        tot_com += n_com
        tot_cad += n_cad
        detalle = f"{n_com} comentarios"
        if args.docstrings:
            detalle += f", {n_cad} docstrings/cadenas"
        print(f"  {'limpiado' if escribir else 'se limpiaría'}: {rel}  ({detalle})")
        if escribir:
            if args.backup:
                shutil.copy2(ruta, ruta.with_name(ruta.name + ".bak"))
            ruta.write_bytes(nuevo)

    verbo = "eliminados" if escribir else "a eliminar"
    resumen = f"\n{n_arch} archivos analizados, {n_mod} con cambios, {tot_com} comentarios {verbo}"
    if args.docstrings:
        resumen += f", {tot_cad} docstrings/cadenas {verbo}"
    print(resumen + ".")
    if avisos:
        print("\nAvisos:")
        for a in avisos:
            print("  - " + a)
    if args.output:
        print(f"\nCopia limpia en: {destino}")


if __name__ == "__main__":
    main()
