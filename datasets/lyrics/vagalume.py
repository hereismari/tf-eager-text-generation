'''
    Code originally from: 
        https://github.com/gabrielgoncalves95/vagalume_crawler
    Modified by: @mari-linhares
'''

# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
from collections import deque
import nltk
import time

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--band-name', type=str, required=True)
parser.add_argument('--language', type=str, required=True, choices=["spanish", "english", "italian", "portuguese", "french"])


REPLACEMENTS = {
    "â‚¬" : "€", "â€š" : "‚", "â€ž" : "„", "â€¦" : "…", "Ë†"  : "ˆ",
    "â€¹" : "‹", "â€˜" : "‘", "â€™" : "’", "â€œ" : "“", "â€"  : "”",
    "â€¢" : "•", "â€“" : "–", "â€”" : "—", "Ëœ"  : "˜", "â„¢" : "™",
    "â€º" : "›", "Å“"  : "œ", "Å’"  : "Œ", "Å¾"  : "ž", "Å¸"  : "Ÿ",
    "Å¡"  : "š", "Å½"  : "Ž", "Â¡"  : "¡", "Â¢"  : "¢", "Â£"  : "£",
    "Â¤"  : "¤", "Â¥"  : "¥", "Â¦"  : "¦", "Â§"  : "§", "Â¨"  : "¨",
    "Â©"  : "©", "Âª"  : "ª", "Â«"  : "«", "Â¬"  : "¬", "Â®"  : "®",
    "Â¯"  : "¯", "Â°"  : "°", "Â±"  : "±", "Â²"  : "²", "Â³"  : "³",
    "Â´"  : "´", "Âµ"  : "µ", "Â¶"  : "¶", "Â·"  : "·", "Â¸"  : "¸",
    "Â¹"  : "¹", "Âº"  : "º", "Â»"  : "»", "Â¼"  : "¼", "Â½"  : "½",
    "Â¾"  : "¾", "Â¿"  : "¿", "Ã€"  : "À", "Ã‚"  : "Â", "Ãƒ"  : "Ã",
    "Ã„"  : "Ä", "Ã…"  : "Å", "Ã†"  : "Æ", "Ã‡"  : "Ç", "Ãˆ"  : "È",
    "Ã‰"  : "É", "ÃŠ"  : "Ê", "Ã‹"  : "Ë", "ÃŒ"  : "Ì", "ÃŽ"  : "Î",
    "Ã‘"  : "Ñ", "Ã’"  : "Ò", "Ã“"  : "Ó", "Ã”"  : "Ô", "Ã•"  : "Õ",
    "Ã–"  : "Ö", "Ã—"  : "×", "Ã˜"  : "Ø", "Ã™"  : "Ù", "Ãš"  : "Ú",
    "Ã›"  : "Û", "Ãœ"  : "Ü", "Ãž"  : "Þ", "ÃŸ"  : "ß", "Ã¡"  : "á",
    "Ã¢"  : "â", "Ã£"  : "ã", "Ã¤"  : "ä", "Ã¥"  : "å", "Ã¦"  : "æ",
    "Ã§"  : "ç", "Ã¨"  : "è", "Ã©"  : "é", "Ãª"  : "ê", "Ã«"  : "ë",
    "Ã¬"  : "ì", "Ã­"   : "í", "Ã®"  : "î", "Ã¯"  : "ï", "Ã°"  : "ð",
    "Ã±"  : "ñ", "Ã²"  : "ò", "Ã³"  : "ó", "Ã´"  : "ô", "Ãµ"  : "õ",
    "Ã¶"  : "ö", "Ã·"  : "÷", "Ã¸"  : "ø", "Ã¹"  : "ù", "Ãº"  : "ú",
    "Ã»"  : "û", "Ã¼"  : "ü", "Ã½"  : "ý", "Ã¾"  : "þ", "Ã¿"  : "ÿ"
}

def replace_all(texto):
    for k in REPLACEMENTS:
        texto = texto.replace(k, REPLACEMENTS[k])
    return texto

def detecta_idioma(texto_para_detectar: str):
    languages = ["spanish","english","italian","portuguese","french"]
    tokens = nltk.tokenize.word_tokenize(texto_para_detectar)
    tokens = [t.strip().lower() for t in tokens]
    lang_count = {}

    for lang in languages:
        stop_words = nltk.corpus.stopwords.words(lang)
        lang_count[lang] = 0

        for word in tokens:
            if word in stop_words:
                lang_count[lang] += 1

    return max(lang_count, key=lang_count.get)


def space_replace(texto: str, item :str):
    texto = texto.replace(item, " ")

    return texto

def main(args):
    url = 'www.vagalume.com.br'
    band = args.band_name
    links = '/%s/' % band
    a = requests.get("http://" + url + links)
    data = a.text
    soup = BeautifulSoup(data)
    songs = deque([])
    file_name = '%s.txt' % band
    arquivo = open(file_name, 'w')

    for link in soup.find_all('a'):
        songs.append(link.get('href'))
    for i in range(0, len(songs)):
        song = songs.pop()
        if (song.find(links)!=-1):        
            songs.append(song)
    for i in range(0, len(songs)):
        song = songs.popleft()
        if (song.find(links)!=-1):        
            songs.append(song)

    remove=[]
    for i in range(0, len(songs)):
        song = songs[i]
        if (song.find('#play')!=-1): 
            remove.append(songs[i])

    for i in range(0, len(remove)):
        songs.remove(remove[i])

    remove.clear()

    for i in range(0, len(songs)):
        song = songs[i]
        if (song.find('traducao')!=-1): 
            remove.append(songs[i])

    for i in range(0, len(remove)):
        songs.remove(remove[i])

    remove.clear()


    for i in range(0, len(songs)):
        song = songs[i]
        if (song.find('cifrada')!=-1): 
            remove.append(songs[i])

    for i in range(0, len(remove)):
        songs.remove(remove[i])

    remove.clear()

    for i in range(0, int(len(songs)/2)):
        song = songs[i]
        if (song.find('/news/')!=-1): 
            remove.append(i)
        if (song.find('/tags/')!=-1): 
            remove.append(i)
        if (song.find('/popularidade/')!=-1): 
            remove.append(i)
        if (song.find('/fotos/')!=-1): 
            remove.append(i)

    try:
        index = max(remove, key=int)
    except:
        index = 0

    for i in range(0, index + 1):
        try:
            del songs[0]
        except:
            continue

    remove.clear()

    for i in range(int(len(songs)/2), int(len(songs))):
        song = songs[i]
        if (song.find('/news/')!=-1): 
            remove.append(i)
        if (song.find('/fotos/')!=-1): 
            remove.append(i)
        if (song.find('/popularidade/')!=-1): 
            remove.append(i)
        if (song.find('/discografia/')!=-1): 
            remove.append(i)

    try:
        index = min(remove, key=int)
    except:
        index = len(songs)

    for i in range(index, len(songs)):
        try:
            del songs[len(songs)-1]

        except:
            continue

    contador = 0
    for musica in songs:
        print("[x] " + str(musica))
        s = requests.get("http://" + url + str(musica))
        contador = contador + 1
        
        if contador > 200:
            time.sleep(60)
            contador = 0 

        data = s.text
        soup = BeautifulSoup(data)
        letras = deque([])
        letra = soup.find("div", id="lyrics")
        try:
            texto = str(letra.get_text(separator="\n"))
        except:
            continue

        char_list_to_nothing = ["{refrão}", "(2x)", "(3x)", "(1x)", "(4x)", "?", "!", ",", ".", ":", "(2 vezes)", "(1ª vez)", "(2ª vez)", "(", ")", "[", "]", "/", "'", '"', "[cr]",
                        "[repete tudo]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "--", "_"]
        char_list_to_space = ["  ", "   ", "    ", "     ", "      ", " x "]
 
        for item in char_list_to_nothing + char_list_to_space:
            texto = space_replace(texto, item)

        texto = texto.replace("noix", "nós")
        texto = texto.replace("vc", "você")
        texto = texto + '\n'

        texto = replace_all(texto)

        idioma = detecta_idioma(texto)

        if(idioma != args.language):
            print ("Error: invalid language detected, song will be ignored.")
            continue
        try:
            arquivo.write(str(texto))
        except:
            print ("Some error ocurred song will be ignored.")

    #print(songs)
    arquivo.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
