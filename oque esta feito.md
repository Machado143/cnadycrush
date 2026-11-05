```
oque esta feito
```

```
quero fazer um bot para jogar candy crush
```

```
Conclusão: é viável. A abordagem mais prática é visão computacional + automação
de entrada, rodando o jogo em emulador Android (ADB) ou no navegador. Um bot
básico (greedy) leva de alguns dias a poucas semanas. Um bot que cumpre
objetivos de fase de forma confiável é bem mais trabalhoso.
```

```
Suposições: você quer automatizar o jogo real, não criar um clone. Não sei
plataforma nem linguagem; assumo Python.
```

```
Arquitetura
```

```
EtapaComo
Captura de telaadb exec-out screencap -p (Android) ou mss (PC)
Detecção do gridOpenCV: localizar a área do tabuleiro, dividir em células
(ex.: 9x9)
Classificação de peçasCor média/template matching por célula; doces especiais
e bloqueadores exigem templates próprios
Busca de jogadasSimular todas as trocas adjacentes e verificar match de 3+
Pontuação da jogadaHeurística: tamanho do match, criação de doces
especiais, objetivo da fase
Execuçãoadb shell input swipe x1 y1 x2 y2 ou pyautogui
SincronizaçãoEsperar a animação/cascata terminar (comparar frames até
estabilizar) antes de ler de novo
Pontos críticos
```

```
Reposição aleatória: não dá para prever as peças que caem. Lookahead profundo
tem pouco valor; 1 a 2 níveis bastam.
Objetivos variam por fase (geleia, ingredientes, ordens, bloqueadores). A
heurística precisa ser trocada por fase, e esse é o maior custo do projeto.
Detecção instável: efeitos visuais, brilhos e animações causam leitura errada.
Esse costuma ser o gargalo real, não o algoritmo.
Vidas e anúncios: o bot precisa lidar com pop-ups, fim de vidas e telas de
transição, ou para sozinho.
Riscos
```

```
Automação viola os termos de uso da King. Há risco de banimento de conta,
principalmente com ações em intervalos regulares. Fato: os termos proíbem
automação; a frequência de banimento eu não sei estimar.
Evite ler memória ou interceptar tráfego do app. É mais complexo, mais arriscado
e não vou ajudar com isso.
Alternativa: se o objetivo é aprender ou pesquisar, é melhor implementar um
ambiente match-3 próprio e treinar um agente (busca ou RL). Isso elimina visão
computacional, risco de ban e problemas de sincronização.
```

```
Núcleo de lógica (busca de jogadas):
```

```
python
def find_matches(g):
    n, m = len(g), len(g[0])
    hit = set()
    for i in range(n):
        for j in range(m):
            c = g[i][j]
            if c is None: continue
            if j+2 < m and g[i][j+1] == c == g[i][j+2]:
                hit |= {(i,j),(i,j+1),(i,j+2)}
            if i+2 < n and g[i+1][j] == c == g[i+2][j]:
                hit |= {(i,j),(i+1,j),(i+2,j)}
    return hit
def best_move(g):
```

```
    best, best_score = None, 0
    for i in range(len(g)):
        for j in range(len(g[0])):
            for di, dj in ((0,1),(1,0)):
                a, b = i+di, j+dj
                if a >= len(g) or b >= len(g[0]): continue
                g[i][j], g[a][b] = g[a][b], g[i][j]
                s = len(find_matches(g))
                g[i][j], g[a][b] = g[a][b], g[i][j]
                if s > best_score:
                    best, best_score = ((i,j),(a,b)), s
    return best
Perguntas:
```

```
Plataforma: Android (emulador ou aparelho real), PC/Facebook ou iOS?
Objetivo: jogar o jogo real ou aprender/pesquisar (RL, visão)?
Linguagem preferida, se não for Python?
compensa mais eu baixar no pc ou usar no celular
```

```
Conclusão: para desenvolver e rodar o bot, PC com emulador Android + ADB é a
melhor opção. Celular real via ADB é a segunda. Versão nativa de PC (Microsoft
Store/navegador) é a mais simples, mas a mais limitada.
```

```
OpçãoVantagensDesvantagens
Emulador no PC (ADB)Resolução fixa e configurável; captura e toque
programáticos sem ocupar o mouse; roda em segundo plano; fácil depurarExige PC
com virtualização; emulador pode ser detectado (não sei se a King bloqueia);
pode ter pior desempenho em PCs fracos
Celular real (ADB via USB/Wi-Fi)Ambiente idêntico ao de um jogador comum;
sem emuladorAparelho fica ocupado; screencap costuma ser mais lento
(centenas de ms por frame); depuração mais trabalhosa
Versão nativa PC (Store/navegador)Setup mais simples (mss + pyautogui)
Controla mouse e foco da janela, então você não usa o PC; sem ADB;
disponibilidade e versão atual da Store/Facebook eu não verifiquei
Pontos que afetam a decisão
```

```
Progresso e conta: o progresso sincroniza por conta (King/Facebook). Se usar a
conta principal, o risco de banimento vale para ela em qualquer opção. Fato:
automação viola os termos. Considere uma conta separada para testes.
Windows Subsystem for Android: foi descontinuado pela Microsoft em 2025 (certeza
média, confirme). Use BlueStacks, LDPlayer ou o AVD do Android Studio.
Detecção de emulador: não tenho dado confiável sobre isso para o Candy Crush.
Trate como risco desconhecido.
```

```
Desempenho do bot: o gargalo é a sincronização de animação, não a plataforma.
Emulador com ADB dá latência menor e mais previsível que celular real.
Recomendação: comece com emulador (AVD ou LDPlayer) em resolução fixa, por ser o
ambiente mais reprodutível para calibrar a detecção do grid. Migre para celular
real só se houver problema com o emulador.
```

# `Perguntas:` 

```
Você tem PC com virtualização habilitada e RAM suficiente (8 GB ou mais)?
Tem um celular Android sobrando, ou só o de uso diário?
Pretende usar sua conta principal ou uma de teste?
```

```
Conclusão: funciona. É o setup mais simples, mas ocupa mouse e tela enquanto o
bot roda. Não verifiquei se a versão da Microsoft Store ou do navegador está
disponível e atualizada; confirme ao baixar.
```

```
Stack: Python, mss (captura), opencv-python + numpy (visão), pyautogui (arrasto
do mouse), keyboard (tecla de parada).
```

# `Configuração inicial` 

```
Fixe o tamanho da janela e não mude depois. Toda a calibração do grid depende
```

```
disso.
A janela precisa ficar visível, em primeiro plano e sem sobreposição. Minimizada
ou coberta, a captura falha.
Escala do Windows diferente de 100% desalinha coordenadas. Mitigação: chamar
SetProcessDPIAware() no início do script ou usar 100%.
Se houver vários monitores, descubra o offset do monitor onde o jogo está,
porque mss e pyautogui usam coordenadas globais.
Captura e movimento (base):
```

```
python
import ctypes, time
import mss, numpy as np, pyautogui
```

```
ctypes.windll.user32.SetProcessDPIAware()
pyautogui.FAILSAFE = True   # mouse no canto superior esquerdo aborta
pyautogui.PAUSE = 0.05
```

```
# região do tabuleiro: calibrar uma vez (x, y, largura, altura)
BOARD = {"left": 400, "top": 200, "width": 540, "height": 540}
ROWS = COLS = 9
```

```
def grab():
    with mss.mss() as s:
```

```
        return np.array(s.grab(BOARD))[:, :, :3]  # BGR
```

```
def cell_center(r, c):
```

```
    cw, ch = BOARD["width"] / COLS, BOARD["height"] / ROWS
    return (BOARD["left"] + (c + .5) * cw, BOARD["top"] + (r + .5) * ch)
```

```
def swap(a, b):
```

```
    x1, y1 = cell_center(*a)
    x2, y2 = cell_center(*b)
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=0.15, button="left")
Os valores de BOARD e o tamanho do grid são placeholders. Muitas fases não têm
grade 9x9 nem células todas ativas.
```

```
Cuidados
```

```
Sincronização: depois de cada jogada, capture frames em loop até dois
consecutivos serem quase idênticos (diferença média abaixo de um limiar). Sem
isso, o bot lê o tabuleiro no meio da cascata.
Parada de emergência: o mouse está sob controle do script. Mantenha FAILSAFE
ligado e adicione uma tecla global (keyboard.add_hotkey("esc", ...)) para
encerrar.
Pop-ups e telas de transição: detecte por template matching e trate (fechar,
seguir para a próxima fase). Sem isso o bot trava.
Intervalos: jogadas em ritmo perfeitamente regular são o padrão mais fácil de
detectar. Isso é inferência minha, não dado da King. Adicionar variação não
elimina o risco de banimento, que continua existindo.
Dados antes do código: salve capturas de tela de várias fases e desenvolva a
classificação de peças offline, sem o jogo aberto. Isso acelera a depuração.
Ordem recomendada
```

```
Capturar e recortar o tabuleiro.
Classificar cada célula (cor média ou template).
Imprimir a matriz no terminal e conferir com a tela.
Só então ligar a busca de jogadas e o swap.
Perguntas:
```

```
Versão da Microsoft Store ou jogo no navegador?
Sua tela tem escala diferente de 100% ou mais de um monitor?
Confirming the cause before running a small test
Resultado: a matriz bate com a imagem em todas as 64 células (0 = verde, 1 =
```

```
azul, 2 = laranja, 3 = roxo). A causa era a paleta aprendida com outra imagem na
tela, como suspeitado. A leitura está validada para esse tabuleiro.
```

```
Limite da validação: foi testado em um único tabuleiro, só com doces simples.
Ainda não foi testado com doces especiais, bloqueadores nem durante cascatas.
```

```
Próximo passo: testar run com poucas jogadas
```

```
python candybot.py run --max-moves 3
```

```
Observe e não mexa no mouse durante o teste. Para interromper: ESC, ou mouse no
canto superior esquerdo da tela.
```

```
O que verificar
```

```
O jogo aceita o arrasto? Se o terminal imprimir “tabuleiro não mudou”, o jogo
ignorou o gesto. Nesse caso, a troca precisa ser feita por clique em uma célula
e depois na outra. Eu ajusto swap.
```

```
A jogada escolhida é válida? Compare a jogada impressa (linha, coluna) com a
tela.
```

```
A espera pela cascata funciona? Se o bot jogar enquanto os doces ainda estão
caindo, ele lê o tabuleiro errado. Esse é o sintoma a observar. Se ocorrer,
ajusto STABLE_DIFF e min_wait.
Esperado nos próximos testes
```

```
A paleta vale para o conjunto de cores atual. Se a fase mudar de cores, rode
learn de novo com o tabuleiro visível.
Doces listrados e embrulhados têm cor média parecida com a do doce base. O bot
provavelmente os lê como doce comum, o que é aceitável nesta etapa. Bloqueadores
devem sair como ? e o bot não os move.
Risco: o aviso anterior continua valendo. Automação viola os termos da King e
pode levar a banimento da conta.
```

```
Perguntas:
```

```
O jogo aceitou o arrasto nas 3 jogadas?
Qual foi a saída do terminal ao final do teste?
1793886757264_image.png
1793886762358_image.png
1793886766905_image.png
(ia) PS C:\Users\ifsp\Downloads> python candybot.py run --max-moves 3
Iniciando em 3 s. Deixe o jogo visível e em primeiro plano. ESC para parar.
Jogada 1: (4, 4) -> (4, 5)
Jogada 2: (3, 2) -> (4, 2)
Jogada 3: (2, 3) -> (3, 3)
Fim. 3 jogadas.
(ia) PS C:\Users\ifsp\Downloads>
```

```
Pesquisou na web, editou um arquivo
Sorting the candy images for a learning game.
```

```
Identifying green fish-shaped candy boosters.
```

```
Identificando doces listrados e embrulhados nas imagens.
Avaliando regras de combinação dos doces especiais.
Montando um classificador de doces com exemplos rotulados.
Checking which game the fish candy belongs to.
Narrowing down which candy game the screenshot is from.
Planning a labeled-crop classifier for candy pieces.
```

```
Writing the classifier and collection code for the candy pieces.
```

```
Writing the dedupe and crop-feature helpers for the classifier.
```

```
Deciding which crops the collector saves.
```

```
Building the labeling tool for crop images.
```

```
Adding the on-screen label display and classifier output to the show command.
```

```
Adding a per-cell kind printout and a crop harvester for new labels.
```

```
Wiring the harvester into the run and collect commands.
```

```
Writing the label loop and its key handling.
```

```
Checking the label tool’s keys and the run-command edits.
```

```
Run: opção --collect que guarda recortes de células ainda não reconhecidas
candybot.py
```

```
**Resumo:** você está construindo um bot de Candy Crush em Python para PC, lendo
a tela e arrastando o mouse. A leitura de doces simples foi validada. O foco
atual é treinar um classificador para doces especiais. Hierarquia de pontuação e
RL ficam para depois.
```

# `### Ambiente` 

```
- Windows, venv `(ia)`, pasta `C:\Users\ifsp\Downloads`.
```

- `Tela dividida: VS Code à esquerda e jogo à direita.` 

- `Dependências: `mss`, `numpy`, `opencv-python`, `pyautogui`, `keyboard`. O classificador novo usa só numpy.` 

- ``config.json`: grade 8x8 e paleta de 4 cores (verde, laranja, azul, roxo). A área do tabuleiro foi regravada pelo seu último `seed`.` 

# `### Estado validado` 

- ``calibrate`, `learn` e `show` funcionaram. A matriz bateu com a tela nas 64 células, só com doces simples.` 

```
- `run --max-moves 3` fez 3 jogadas e terminou sem erro. Você confirmou que “deu
certo”.
- Limite dessa validação: um único tabuleiro, sem especiais, sem bloqueadores e
sem cascatas testadas.
```

# `### Decisões` 

- `Aprendizado primeiro, desempenho depois.` 

- `Sem RL no jogo real: não há simulador, cada amostra é lenta e a reposição é aleatória. RL só faz sentido num ambiente match-3 próprio, com especiais e cascatas implementados. Isso é um projeto à parte.` 

```
- A identificação usa aprendizado supervisionado (k-NN sobre recortes
rotulados).
```

# `### Código atual` 

```
O `candybot.py` em `/mnt/user-data/outputs` é a versão mais recente. O `run` não
mudou. A versão com classificador citada no seu log não estava disponível e foi
reescrita.
```

```
ComandoFunção`calibrate`Seleciona a área do tabuleiro e a grade`learn --k
N`Aprende a paleta de cores por k-means`show`Lê o tabuleiro e salva
`debug.png``seed`Seleciona a área do jogo, reaprende a paleta e rotula normais
```

```
em grupos por cor. Opções: `--k --rows --cols --monitor --
delay``collect`Passivo. Salva recortes não reconhecidos enquanto você joga.
Opções: `--seconds --interval --dedupe``label`Rotula por tecla: tipo, depois
cor`train`Gera `model.npz` e mostra acerto por classe`kinds`Imprime o tipo de
cada célula`run`Executa o bot. ESC ou mouse no canto superior esquerdo aborta
```

```
**Classes de rótulo:** `normal_<cor>`, `listrado_vert_<cor>`,
`listrado_horiz_<cor>`, `embrulhado_<cor>`, `peixe_<cor>`, `bomba`,
`bloqueador`, `vazio`. Cores por nome: 1 vermelho, 2 laranja, 3 amarelo, 4
verde, 5 azul, 6 roxo. Listrado é rotulado pela aparência: `v` = listras
verticais na imagem, `h` = horizontais.
```

```
**Método:** recorte 24x24 (BGR) mais magnitude de borda. k-NN com k=3 ponderado.
Se o vizinho mais próximo passar do limite, a célula vira `?` e o bot não a
move.
```

# `### Amostras de especiais que você enviou` 

```
Listrado azul (listras verticais), embrulhado azul, bomba de cor e peixe azul.
Cada recorte tem cerca de 60 px.
```

# `### Fluxo em andamento` 

`1. `seed`: feito por você. Não sei quantos grupos apareceram nem quantos foram pulados.` 

`2. `train`.` 

`3. `collect --seconds 300`, jogando à mão em fase com especiais.` 

`4. `label`.` 

`5. `train` de novo.` 

`6. `kinds` com especiais no tabuleiro, comparando com a tela.` 

# `### Próximas etapas` 

```
1. Hierarquia de pontuação no `best_move`. Valores iniciais são pesos a
calibrar, não oficiais. Criação: 4 em linha gera listrado, L/T/+ gera
embrulhado, 5 em linha gera bomba de cor. Combos em ordem de prioridade:
bomba+bomba, bomba+embrulhado, bomba+listrado, embrulhado+listrado,
listrado+listrado, embrulhado+embrulhado. Confirme as regras no jogo.
```

```
2. Simulador que modele especiais e cascatas. É o pré-requisito para RL.
```

```
3. Calibrar o desempenho no jogo real, com pesos por objetivo de fase.
```

# `### Limitações e riscos` 

- `Só testei compilação e o pipeline de treino e predição, com seus 4 recortes mais variações sintéticas. O acerto de 100% nesse teste não vale como medida. A janela de seleção e a rotulagem por tecla não foram testadas aqui.` 

```
- `--dedupe 2.0` é um valor chutado.
```

```
- Se o k-NN confundir listrado com normal, o próximo passo é uma CNN pequena.
- Não verifiquei se a versão do jogo no seu PC está atual nem se existe detecção
de automação.
```

```
- Automação viola os termos da King e pode levar a banimento. Uma conta separada
reduz o dano, mas não elimina o risco. O `collect` é passivo e não move o mouse.
- O `seed` só vale com tabuleiro sem especiais nem bloqueadores. Grupos
misturados indicam `--k` pequeno.
```

# `### Pendências suas` 

`1. Saída do `train` e quantos grupos o `seed` mostrou.` 

`2. As fases usadas para coletar têm outra grade ou células inativas?` 

`3. Os peixes eram só azuis ou de outras cores também?` 

