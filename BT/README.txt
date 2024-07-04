abstruct, thesisのコンパイルはVScodeを用いてWSL:Ubuntu上で行っており，コマンドの省略可のためにsettings.jsonファイルで設定している．

settings.jsonの中身は以下のようになっている．

-------------------------------------------------------------------------------
    {"latex-workshop.latex.tools": [
        {
            "command": "latexmk",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-pdf",
                "%DOC%"
            ],
            "name": "latexmk"
        },
        {
            "command": "pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ],
            "name": "pdflatex",
            },
            {
                "command": "bibtex",
                "args": [
                    "%DOCFILE%"
                ],
                "name": "bibtex",
            },
            {
                "command": "ptex2pdf",
                "args": [
                    "-interaction=nonstopmode",
                    "-l",
                    "-ot",
                    "-kanji=utf8 -synctex=1",
                    "%DOC%.tex"
                ],
                "name":"ptex2pdf",
            },
           {
                "command": "ptex2pdf",
                "args": [
                    "-l",
                    "-u",
                    "-ot",
                    "-kanji=utf8 -synctex=1",
                    "%DOC%"
                ],
                "name":"ptex2pdf (uplatex)",
            },
            {
                "command": "pbibtex",
                "args": [
                    "-kanji=utf8",
                    "%DOCFILE%"
                ],
                "name": "pbibtex",
            }
     ],
     "latex-workshop.latex.recipes": [
     {
        "name": "ptex2pdf",
        "tools": [
            "ptex2pdf"
        ]
     },
     {
        "name": "ptex2pdf -> pbibtex -> ptex2pdf*2",
        "tools": [
            "ptex2pdf",
            "pbibtex",
            "ptex2pdf",
            "ptex2pdf"
        ]
     },
     {
        "name": "pdflatex -> bibtex -> pdflatex*2",
        "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
        ]
     },
     {
        "name": "latexmk",
        "tools": [
            "latexmk"
        ]
     },
     {
        "name": "pdflatex",
        "tools": [
            "pdflatex"
        ]
     },
     {
        "name": "ptex2pdf (uplatex)",
        "tools": [
            "ptex2pdf (uplatex)"
        ]
     },
     {
        "name": "ptex2pdf (uplatex) -> pbibtex -> ptex2pdf (uplatex) *2",
        "tools": [
            "ptex2pdf (uplatex)",
            "pbibtex",
            "ptex2pdf (uplatex)",
            "ptex2pdf (uplatex)"
        ]
     },
     ],
     
     
     "latex-workshop.latexindent.path": "latexindent.exe",
     "latex-workshop.view.pdf.viewer": "tab",
     
     "[tex]": {
        // スニペット補完中にも補完を使えるようにする
        "editor.suggest.snippetsPreventQuickSuggestions": false,
        // インデント幅を2にする
        "editor.tabSize": 2
     },
     
     "[latex]": {
        // スニペット補完中にも補完を使えるようにする
        "editor.suggest.snippetsPreventQuickSuggestions": false,
        // インデント幅を2にする
        "editor.tabSize": 2
     },  
     
     "[bibtex]": {
        // インデント幅を2にする
        "editor.tabSize": 2
     }
}

-------------------------------------------------------------------------------
Pythonはインターネットからダウンロードできるのでローカルで環境構築しておくこと．
できない場合はgoogle  colaboratoryでじっこうすること．
コードを実行するときに使用するファイルのディレクリーに気をつけること．
ローカルで環境構築した場合はライブラリをダウンロードしておくこと．


