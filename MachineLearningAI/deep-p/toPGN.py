import MySQLdb
import pdb

pdb.set_trace()
# db = MySQLdb.connect("localhost","root","password","challengeChess")
db = MySQLdb.connect("mydbinstance.c4qmkaf8x5fu.us-west-2.rds.amazonaws.com","root","byucs428","challengeChess")
cursor = db.cursor()

sqlQuery = "SELECT id FROM Game"

cursor.execute(sqlQuery)
results = cursor.fetchall()

games = []

for res in results:
	games.append(res[0])

with open("./testPGN.pgn", 'w+') as of:
	for g_id in games:
		sqlQuery = "SELECT from_position, to_position, name from move WHERE game_id = %s" % (g_id)
		cursor.execute(sqlQuery)
		results = cursor.fetchall()
		p.set_trace()
		output = ""
		counter = 1
		for res in results:
			if counter % 2 is 1:
				output += str(counter) + ". "

			counter += 1
			piece = str(res[2])
			from_loc = str(res[0])
			to_loc = str(res[1])
			if not 'P' in piece:
				move_str = piece+from_loc+to_loc
				output += move_str
				of.write(output)
			else:
				move_str = from_loc+to_loc
				output += move_str
				of.write(output)


		

