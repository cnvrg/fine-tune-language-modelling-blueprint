import pandas as pd
import unittest
from finetune import DataTrainingArguments, get_dataset
import os
import pathlib as pl
import transformers
from transformers import PreTrainedTokenizer, LineByLineTextDataset, TextDataset, AutoTokenizer

print(os.getcwd())

class TestFT(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.train_data = [
            [
                "<BOS> <action> Prince (Vivek Oberoi) is a sharp and intelligent burglar, but when he awakes one morning, he finds that he does not remember anything about his past. He goes to a club and meets a girl who claims to be his girlfriend Maya. The next day, he meets a second girl who claims to be his girlfriend named Maya. She also claims that they work for the police and are after a man named Sarang. She reveals that they must find a special coin and give it to Sarang, after which the cops will arrest him. This coin has a chip in it that can go into one's mind and change one's thoughts completely.  They find the coin inside Prince's shoe and give it to Sarang. Prince finds out that the second 'Maya' is really a woman named Serena who works for Sarang. His servant P.K. works for Sarang along with her. The coin is revealed to be a fake. Just as Prince is trapped, the real Maya, who is in fact his girlfriend, saves him and tells him the actual story. The chip was put inside Prince so that he could work for Sarang. The two began running away. The chip made Prince's brain like a computer, so that once he woke up from sleep, he would forget everything.  Prince and Maya meet Sarang Chinea, who tells them a side effect: every morning when Prince wakes up, his brain crashes, which results in a lot of pain. He has only six days to live as a result. This particular day was the last day. Prince and Maya find the coin, which can save Prince, and Serena goes after them, leading to a high-profile chase. Prince manages to keep the coin safe but faints. A mysterious car arrives, pulls him in, and flees. Shortly after, Maya gets a phone call, saying she must come to a certain location if she wants Prince alive. The caller is a friend of Prince's named Mike. It is revealed that the first Maya that Prince had met at the club, is actually named Priya. Then Maya calls the police.  Priya takes the coin and runs away with it, but Mike tells Maya that he has the real coin; the one Priya fled with is a fake. While they are fixing an unconscious Prince, Priya returns, begging to be saved, but gets shot by Sarang and his gang. Prince awakes from his deep sleep. Sarang and his gang escape, but Prince puts a tracking device on Sarang. They track him down, and Prince and Sarang engage in a brief fight. Prince ultimately gets the coin and Sarang falls off a waterfall to his death. While Prince and Maya are looking forward to a happily-ever-after, in a twist ending, Serena's eyes open, implying that she is still alive. <EOS>"
            ],
            [
                "<BOS> <drama> Life is going along smoothly for Jeff and Mari Thompson but not for any other couple they know, or so it seems. Everyone they know is getting divorced.  Their life is disrupted when Mari's old college friend, Barbara, comes into it and begins a fling with Jeff, which causes Mari to contemplate an affair of her own. <EOS>"
            ],
            [
                "<BOS> <drama> A young girl suddenly finds herself wealthy, but lacking in social graces. She calls upon the disinherited son from a wealthy family for help. <EOS>"
            ],
            [
                "<BOS> <drama> Varghese (Mohanlal) is a Kalari master. Shyam (Vineeth) is his favourite disciple. He comes to his master declaring his love for Maya (Urmila Matondkar). He says that her parents have kept her under house arrest and he needs his master's help to rescue her.  Varghese kidnaps Maya and brings her to Shyam but Shyam tries to kill her. Varghese understands that Shyam is not her lover and she is the only witness in another homicide which involves Shyam. But Maya tells Varghese that Shyam is innocent. Varghese tries to help both of them. <EOS>"
            ],
            [
                "<BOS> <drama> Liz is a Los Angeles street prostitute. The audience first sees her attempting to get a customer on a busy downtown street near a tunnel. She addresses the audience directly on her life and problems throughout the film. When a van stops by, she gives it the brush off, recalling the last time she serviced a man in a van: it turned out there were several other men in the van, who gang-raped her and left her for dead. A passerby gives her his handkerchief and offers to take her to a hospital. She refuses, makes up a boyfriend story and asks for some money. She sends him the money back with a thank you note and a new handkerchief.  Liz isn't merely attempting to get a customer, however: she is attempting to escape her pimp, Blake. Blake is a well-dressed, businesslike and extremely controlling man.  As Liz stops off at a strip club for a drink, she explains how she ended up as she did: she was a small town girl, who married a violent drunk named Charlie (Frank Smith). Though they have a child together, she can no longer take it and leaves him, taking her son with her, as he's sleeping it off. She takes a job on the graveyard shift at a diner, and when a customer offers her more money to have sex with him, she decides, given her rather low pay, to take it. She does this independently for a time until she meets Blake, who takes her to LA. Though Blake does do some things for her (including getting her tattooed), he is ultimately as cruel as her husband, so she decides to escape from him.  A local homeless person/street performer named Rasta decides to treat Liz to a movie. Though Rasta is a bit scary (his act involves walking on broken glass), Liz agrees. At this point the scenes of Liz and Rasta at the movie are intercut with Blake explaining his life to the audience, giving the impression that Liz and Rasta are watching Blake's soliloquy.  After the movie, Liz talks to the audience about her son, whom she clearly loves, though he's now in foster care. She finally gets a customer and services him. He has a heart attack, and Liz panics, trying to give him mouth-to-mouth resuscitation, without success. Blake happens along then. He takes Liz's money and tries to rob the dead customer. When Liz tries to stop him, Blake tries to strangle Liz and threatens to put her son into prostitution, with Liz retorting 'I'll kill you first!'. Rasta comes to the rescue, killing Blake. A grateful Liz gives her thanks and walks away. <EOS>"
            ],
         ]
        self.eval_data = [
            [
                "<BOS> <drama> The film's content is derived from three previously released animated featurettes Disney produced based upon the Winnie-the-Pooh books by A. A. Milne: Winnie the Pooh and the Honey Tree (1966), Winnie the Pooh and the Blustery Day (1968), and Winnie the Pooh and Tigger Too (1974). Extra material was used to link the three featurettes together to allow the stories to merge into each other.  A fourth, shorter featurette was added to bring the film to a close. The sequence was based on the final chapter of The House at Pooh Corner, where Christopher Robin must leave the Hundred Acre Wood behind as he is starting school. In it, Christopher Robin and Pooh discuss what they liked doing together and the boy asks his bear to promise to remember him and to keep some of the memories of their time together alive. Pooh agrees to do so, and the film closes with The Narrator saying that wherever Christopher Robin goes, Pooh will always be waiting for him whenever he returns.<EOS>" 
            ],
            [
                "<BOS> <drama> In bathroom ceramics factory W.C. Boggs & Son, the traditionalist owner W.C. Boggs (Kenneth Williams) is having no end of trouble. Bolshy and lazy union representative Vic Spanner (Kenneth Cope) continually stirs up trouble in the works, to the irritation of his co-workers and management. He calls a strike for almost any minor incident â€“ or because he wants time off to attend a local football match. Sid Plummer (Sid James) is the site foreman bridging the gap between workers and management, shrewdly keeping the place going amid the unrest.  Prissy floral-shirt-wearing product designer Charles Coote (Charles Hawtrey) has included a bidet in his latest range of designs, but W.C. objects to the manufacture of such 'dubious' items. W.C. will not change his stance even after his son, Lewis Boggs (Richard O'Callaghan), secures a large overseas order for the bidets. It is a deal that could save the struggling firm, which W.C. has to admit is in debt to the banks.  Vic's dim stooge Bernie Hulke (Bernard Bresslaw) provides bumbling assistance in both his union machinations and his attempts to woo Sid's daughter, factory canteen worker Myrtle (Jacki Piper). She is torn between Vic and Lewis Boggs, who is something of a playboy but insists he loves her.  Sid's wife is Beattie (Hattie Jacques), a lazy housewife who does little but fuss over her pet budgie, Joey, which refuses to talk despite her concerted efforts. Their neighbour is Sid's brassy and lascivious co-worker Chloe Moore (Joan Sims). Chloe contends with the endless strikes and with her crude, travelling salesman husband Fred (Bill Maynard), who neglects her and leaves her dissatisfied. Chloe and Sid enjoy a flirtatious relationship and are sorely tempted to stray. Unusually for Sid James, his character is a faithful husband, albeit a cheeky and borderline-lecherous one.  Sid and Beattie find that Joey can correctly predict winners of horseraces â€“ he tweets when the horse's name is read out. Sid bets on Joey's tips and makes several large wins â€“ including a vital Â£1,000 loaned to W.C. when the banks refuse a bridging loan â€“ before Sid is barred by Benny (Davy Kaye) his bookie after making several payouts.  The strikers finally return to work, but it is only to attend the annual works outing, a coach trip to Brighton. A good time is had by all with barriers coming down between workers and management, thanks largely to that great social lubricant, alcohol. W.C. becomes intoxicated and spends the day â€“ and it seems the night â€“ with his faithful, adoring secretary, Miss Hortense Withering (Patsy Rowlands). Lewis Boggs manages to win Myrtle from Vic Spanner, giving his rival a beating, and the couple elope. After arriving home late after the outing and with Fred away, Chloe invites Sid in for a cup of tea. They fight their desires and ultimately decide not to have the tea fearing that neighbours might see Sid enter Chloe's home and get the wrong idea.  At the picket lines the next day, Vic gets his comeuppance â€“ partly at the hands of his mother (literally, as she spanks him in public) â€“ and the workers and management all pull together to produce the big order to save the firm. <EOS>"
            ],
            [
                "<BOS> <horror> The Freeling family has sent Carol Anne (Heather O'Rourke) to live with Diane's sister Pat (Nancy Allen) and her husband Bruce Gardner (Tom Skerritt) in Chicago. Carol Anne has been told she is living with her aunt and uncle temporarily to attend a unique school for gifted children with emotional problems, though Pat thinks it is because Steven and Diane just wanted Carol Anne out of their house. Pat and Bruce are unaware of the events that the Freeling family had endured in the previous two films, only noting that Steven was involved in a bad land deal. Along with Donna (Lara Flynn Boyle), Bruce's daughter from a previous marriage, they live in the luxury skyscraper (Chicago's 100-story John Hancock Center) of which Bruce is the manager.  Carol Anne has been made to discuss her experiences by her teacher/psychiatrist, Dr. Seaton (Richard Fire). Seaton believes her to be delusional; however, the constant discussion has enabled the evil spirit of Rev. Henry Kane (Nathan Davis) to locate Carol Anne and bring him back from the limbo he was sent during his previous encounter with her. Not believing in ghosts, Dr. Seaton has come to the conclusion that Carol Anne is a manipulative child with the ability to perform mass hypnosis, making people believe they were attacked by ghosts. Also during this period, Tangina Barrons (Zelda Rubinstein) realizes that Kane has found Carol Anne and travels cross-country to protect her.  That night, Kane drains the high rise of heat and takes possession of reflections in mirrors, causing the reflections of people to act independently of their physical counterparts. When Carol Anne is left alone that night, Kane attempts to use the mirrors in her room to capture her, but she escapes with the help of Tangina, who telepathically tells Carol Anne to break the mirror. Donna and her boyfriend, Scott, see a frightened Carol Anne running through the high rise's parking lot, and move to rescue her; however, before they can, all three are taken to the Other Side through a puddle by Kane and his people. By this point, Tangina and Dr. Seaton are both at the high rise, along with Pat and Bruce. Dr. Seaton stubbornly assumes that Carol Anne has staged the entire thing, while Tangina tries to get her back.  Scott is seemingly released from the Other Side through a pool in the high rise, and Donna reappears after Tangina is taken by Kane disguised as Carol Anne. Scott is left at his home with his parents. Nobody notices that the symbols on Donna's clothing are reversed from what they were before she was taken. As Dr. Seaton attempts to calm Donna, Bruce sees Carol Anne's reflection in the mirror and chases her while Pat follows. Dr. Seaton is not far behind, and he believes he sees Carol Anne in the elevator. However, after Dr. Seaton approaches the elevator doors, Donna appears behind him and pushes him to his death down the empty elevator shaft. At this point it is revealed that what came back was not Donna, but a reflection of her under Kane's control, which then vanishes back into the mirror with a reflection of Scott at its side.  However, even in death Tangina is still more powerful than Kane expects. She returns long enough to give Pat and Bruce her necklace and an important piece of advice. Pat and Bruce struggle to find Carol Anne, but Bruce is captured and eventually Pat is forced to prove her love for Carol Anne in a final face-off against Kane. Tangina manages to convince Kane to go into the light with her. Donna, Bruce, and Carol Anne are returned to Pat. The final scene shows lightning flashing over the building, and Kane's evil laughter is heard. <EOS>"
            ],
         ]

        with open("6_genre_clean_training_data_small_unittests.txt", "w") as output:
            for row in self.train_data:
                for col in row: 
                    output.write(str(col) + '\n')

        with open("6_genre_eval_data_small_unittests.txt", "w") as output:
            for row in self.eval_data:
                for col in row:
                    output.write(str(col) + '\n')


    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))
    
    def assertFileExist(self, path):
        if not pl.Path(path):
            raise ValueError(
                "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                "or remove the --do_eval argument."
            )
    def assertDirExist(self, path):
        if (
            os.path.exists(path)
        ):
            raise ValueError(
                f"Output directory ({path}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

    # Test 1
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("./6_genre_clean_training_data_small.txt")
        self.assertIsFile(file)

    # Test 2
    def test_data_paths(self):
        """Checks if input file existing"""
        file = pl.Path("./6_genre_eval_data_small.txt")
        self.assertIsFile(file)

    # Test 3
    def test_eval_exit(self):
        """Checks if Eval file provided"""
        file = pl.Path("./6_genre_eval_data_small.txt")
        self.assertFileExist(file)

    # Test 4
    def test_outputdir_exist(self):
        """Checks if output_dir existing"""
        file = pl.Path("./story_generator_checkpoint_" + "distilgpt2")
        self.assertDirExist(file)

    # Test 5
    def test_return_type(self):
        """Checks if the function returns the type of transformers.data.datasets.language_modeling.LineByLineTextDataset"""
        data_args = DataTrainingArguments(
            train_data_file='6_genre_clean_training_data_small_unittests.txt',
            eval_data_file='6_genre_eval_data_small_unittests.txt',
            line_by_line=True,
            block_size=256,

            overwrite_cache=True,
        )
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        returned_lists_trainy = get_dataset(data_args, tokenizer=tokenizer)
        returned_lists_evaly = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        self.assertIsInstance(returned_lists_trainy, transformers.data.datasets.language_modeling.LineByLineTextDataset)
        self.assertIsInstance(returned_lists_evaly, transformers.data.datasets.language_modeling.LineByLineTextDataset)
        if data_args.train_data_file and data_args.eval_data_file:
            os.remove(data_args.train_data_file)
            os.remove(data_args.eval_data_file)

if __name__ == "__main__":
    unittest.main()