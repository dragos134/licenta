adr_lex_file = open('ADR_lexicon.tsv', 'r')
adr_lex = adr_lex_file.read()
adr_lex_file.close()


adr_lex = adr_lex.split('###############################################################')[2].split('\n')

print(adr_lex[:100])