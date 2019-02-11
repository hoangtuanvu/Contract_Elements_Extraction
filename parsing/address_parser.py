from parsing.japanese_parser import KANJI_NUMERALS
import MeCab
from itertools import groupby, count
import sys


class AddressParser:
    """Parse input into 2 components like address and other"""

    def __init__(self, input):
        """Take input address"""
        self.mt = MeCab.Tagger(" ".join(sys.argv))
        self.numbers = self.get_numbers()
        self.parse(input)

    def parse(self, input):
        """Deconstruct Japanese address that contains other components"""
        self.mt.parse(input)
        res = self.mt.parseToNode(input)

        cpn_idx = -1
        candidates = []
        all_features = []
        while res:
            cpn_idx += 1
            features = res.feature.split(',')
            if features[0] == "BOS/EOS":
                res = res.next
                continue

            if cpn_idx == 1:
                curr_idx = 0
            else:
                curr_idx = len(input) - len(res.surface)

            all_features.append([cpn_idx, curr_idx, features])
            # print(res.feature)
            if features[2] == "地域":
                candidates.append(cpn_idx)
            res = res.next

        if len(candidates) <= 1:
            self.address = ""
            self.other = input
        else:
            longest_ids = self.get_longest_seq(candidates)
            if len(longest_ids) == 1:
                self.address = ""
                self.other = input
            else:
                if longest_ids[0] != 1:
                    self.other = input[:all_features[longest_ids[0] - 1][1]]
                    self.address = input[all_features[longest_ids[0] - 1][1]:]
                else:
                    split_point = -1
                    for i in range(longest_ids[-1], len(all_features)):
                        if all_features[i][-1][1] == "数":
                            if i == len(all_features) - 1:
                                chars = input[all_features[i][1]:]
                            else:
                                chars = input[all_features[i][1]: all_features[i + 1][1]]

                            for ch in chars:
                                if ch not in self.numbers:
                                    split_point = i
                                    break

                        if all_features[i][-1][1] not in ['数', '一般', '接尾', 'サ変接続', '連体化']:
                            split_point = i
                            break

                    if split_point == -1:
                        self.address = input
                        self.other = ""
                    else:
                        self.address = input[:all_features[split_point][1]]
                        self.other = input[all_features[split_point][1]:]

    @staticmethod
    def get_numbers():
        numbers = []
        for item in list(KANJI_NUMERALS.keys()):
            numbers.append(item)
        for item in list(KANJI_NUMERALS.values()):
            numbers.append(item)

        return numbers

    @staticmethod
    def get_longest_seq(array_input):
        c = count()
        val = max((list(g) for _, g in groupby(array_input, lambda x: x - next(c))), key=len)
        return val

    def get_output_components(self):
        return {
            "address": self.address,
            "other": self.other
        }


if __name__ == "__main__":
    text = "宮崎県都城市九州農政局南部九州土地改良調査管理事務所"
    text = "〒460-8508名古屋市中区三の丸三丁目1番1号"
    addr = AddressParser(text)
    print(addr.get_output_components())
