"""Docutils Markdown parser"""

from collections import OrderedDict

from docutils import parsers, nodes
import html
import markdown
from markdown import util
import os
from pydash import _
import re
import yaml
from lxml import etree
from io import StringIO, BytesIO
from docutils.utils import new_document

__all__ = ['MarkdownParser']

TAGS_INLINE = set("""
b, big, i, small, tt
abbr, acronym, cite, code, dfn, em, kbd, strong, samp, var
a, bdo, br, img, map, object, q, script, span, sub, sup
button, input, label, select, textarea, caption
""".replace(",", "").split())
INVALID_ANCHOR_CHARS = re.compile(
    "[^-_:.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz]")
MAYBE_HTML_TAG = re.compile("<([a-z]+)")


def to_html_anchor(s):
    if not s:
        return s
    if not s[0].isalpha():
        s = "-" + s
    return INVALID_ANCHOR_CHARS.sub("-", s.lower())


IGNORE_ALL_CHILDREN = object()


class Markdown(markdown.Markdown):

    def parse(self, source):
        """
        Like super.convert() but returns the parse tree
        """

        # Fixup the source text
        if not source.strip():
            return ''  # a blank unicode string

        try:
            source = str(source)
        except UnicodeDecodeError as e:  # pragma: no cover
            # Customise error message while maintaining original trackback
            e.reason += '. -- Note: Markdown only accepts unicode input!'
            raise

        # Split into lines and run the line preprocessors.
        self.lines = source.split("\n")
        # newlines = []
        # newlines[:] = [n + "  " for n in self.lines]
        # self.lines = newlines
        for prep in self.preprocessors:
            self.lines = prep.run(self.lines)

        # Parse the high-level elements.
        root = self.parser.parseDocument(self.lines).getroot()

        # Run the tree-processors
        for treeprocessor in self.treeprocessors:
            newRoot = treeprocessor.run(root)
            if newRoot is not None:
                root = newRoot

        # Serialize _properly_.  Strip top-level tags.
        output = self.serializer(root)

        if self.stripTopLevelTags:
            try:
                start = output.index(
                    '<%s>' % self.doc_tag) + len(self.doc_tag) + 2
                end = output.rindex('</%s>' % self.doc_tag)
                output = output[start:end].strip()
            except ValueError:  # pragma: no cover
                if output.strip().endswith('<%s />' % self.doc_tag):
                    # We have an empty document
                    output = ''
                else:
                    # We have a serious problem
                    raise ValueError('Markdown failed to strip top-level '
                                     'tags. Document=%r' % output.strip())

        # Run the text post-processors
        for pp in self.postprocessors:
            output = pp.run(output)

        # CLEAN UP OUTPUT HTML

        # TABLE CAPTIONS
        regex = re.compile(r'<tr>\s*<td>\[(?P<caption>.*?)\](\[(?P<label>.*?)\])?</td>\s*(<td></td>\s*)+</tr>\s*</tbody>', re.S)
        thisIter = regex.finditer(output)
        for item in thisIter:
            thisMatch = item.group()
            thisReplace = f'</tbody>\n<caption style="caption-side: bottom;">{item.group("caption")}</caption>'
            output = output.replace(thisMatch, thisReplace)
            print(output)

        # CONVERT THE HTML BACK TO ROOT
        parser = etree.HTMLParser()
        tree = etree.parse(StringIO(output.strip()), parser)
        root = tree.getroot()

        return root


class MarkdownParser(parsers.Parser):
    """Docutils parser for Markdown"""

    supported = ('md', 'markdown')
    translate_section_name = None

    default_config = {
        'extensions': [],
        'extension_configs': {}
    }

    def __init__(self, config={}):
        self._level_to_elem = {}
        self.config = self.default_config.copy()
        self.config.update(config)

    def parse(self, inputstring, document):
        self.document = document
        self.current_node = document
        try:
            new_cfg = self.document.settings.env.config.markdown_parser_config
            self.config.update(new_cfg)
        except AttributeError:
            pass
        self.setup_parse(inputstring, document)
        frontmatter = self.get_frontmatter(inputstring)

        inputstring = inputstring.replace("{{TOC}}", "1234TOC1234")

        # ALLOW FOR MMD STYLE SUP/SUB-SCRIPTS
        regex = re.compile(r'([!~]*\S)~(\S)([!~]*\n)')
        inputstring = regex.sub(r"\1~\2~\3", inputstring)
        regex = re.compile(r'([!\^]*\S)\^(\S)([!\^]*\n)')
        inputstring = regex.sub(r"\1^\2^\3", inputstring)

        # REPLACE ENV VARIABLES
        regex = re.compile(r'(\{\%(\w+)\%\})')
        matchList = regex.findall(inputstring)
        for m in matchList:
            envVar = os.getenv(m[1].strip(), '')
            inputstring = inputstring.replace(m[0], envVar)

        # ALLOW FOR SPHINX :any: ROLE
        # regex = re.compile(r'\[\[(.*?)\]\]')
        # inputstring = regex.sub(r"£££££\1£££££", inputstring)

        # LINK FIXES - INSIDE NON TEXT TAGS E.G. TABLE CELL
        regex = re.compile(r'\|([^\|\n]*\[[^\]]*\]\(\S*\)[^\|\n]*)\|')
        inputstring = regex.sub(r"| <p>\1</p> |", inputstring)
        regex = re.compile(r'\n\: (.*)')
        inputstring = regex.sub(r"\n: <p>\1</p>", inputstring)

        # ALLOW FOR CITATIONS TO SEMI-WORK (AS FOOTNOTES)
        regex = re.compile(r'\[#(.*?)\]')
        inputstring = regex.sub(r"[^cite\1]", inputstring)

        self.md = Markdown(extensions=self.config.get(
            'extensions'), extension_configs=self.config.get('extension_configs'))

        tree = self.md.parse(self.get_md(inputstring) + "\n")
        self.prep_raw_html()

        # the stack for depth-traverse-reading the markdown AST
        self.parse_stack_r = []
        # the stack for depth-traverse-writing the docutils AST
        self.parse_stack_w = [self.current_node]
        # the stack for determining nested sections
        self.parse_stack_h = [0]
        # index into parse_stack_w used for special cases where enter_* wants
        # to append >1 node (e.g. start_new_section) or pop a node
        self.parse_stack_w_old = 1
        self.walk_markdown_ast(tree)
        # text = self.current_node.pformat()
        # if ":caption: Table of Contents" in text:
        #     print("result:: ==== ")
        #     print(text)
        #     import time
        #     time.sleep(3)
        #     sys.exit(0)
        # print("end result")

        self.finish_parse()

    def get_frontmatter(self, string):
        frontmatter = {}
        frontmatter_string = ''
        frontmatter_regex = re.findall(r'^\s*---+((\s|\S)+?)---+', string)
        if len(frontmatter_regex) and len(frontmatter_regex[0]):
            frontmatter_string = frontmatter_regex[0][0]
        if len(frontmatter_string):
            frontmatter = yaml.safe_load(frontmatter_string)
        return frontmatter

    def get_md(self, string):
        return re.sub(r'^\s*---+(\s|\S)+?---+\n((\s|\S)*)', r'\2', string)

    def attrs_to_dict(self, attrs):
        attrs_dict = {}
        for item in attrs:
            if len(item) == 2:
                attrs_dict[item[0]] = item[1]
        return attrs_dict

    def prep_raw_html(self):
        # code adapted from markedown.core.RawHtmlPostprocessor
        replacements = OrderedDict()
        for i in range(self.md.htmlStash.html_counter):
            html = self.md.htmlStash.rawHtmlBlocks[i]

            if self.isblocklevel(html):
                replacements["<p>%s</p>" %
                             (self.md.htmlStash.get_placeholder(i))] = \
                    html + "\n"
            replacements[self.md.htmlStash.get_placeholder(i)] = html
        self.raw_html = replacements
        if replacements:
            self.raw_html_k = re.compile(
                "(" + "|".join(re.escape(k) for k in self.raw_html) + ")")
        else:
            self.raw_html_k = None

    def isblocklevel(self, html):
        m = re.match(r'^\<\/?([^ >]+)', html)
        if m:
            if m.group(1)[0] in ('!', '?', '@', '%'):
                # Comment, php etc...
                return True
            return self.md.is_block_level(m.group(1))
        return False

    def walk_markdown_ast(self, node):
        try:
            n = node.tag.lower()
        except:
            return
        r_depth = len(self.parse_stack_r)
        self.parse_stack_w_old = len(self.parse_stack_w)

        res = self.dispatch(True, n, node)
        if res is IGNORE_ALL_CHILDREN:
            return
        # shortcut for pushing one item so visitors don't have to
        if res is not None and res != self.parse_stack_w[-1]:
            # add any leftover attributes to the docutils node.
            # this is a slight hack to make attr_list "sort of work" - however
            # docutils interprets attributes in its own way, not as html
            # http://docutils.sourceforge.net/docs/ref/rst/directives.html
            # e.g. style="" usually doesn't work, but some others do by chance
            for (k, v) in node.attrib.items():
                if k not in res:
                    res[k] = v
            self.append_node(res)
        # add text
        if node.text and node.text.strip():
            self.append_text(node.text)

        # dispatch might have modified parse_stack_w_old, so read it again
        w_depth = self.parse_stack_w_old

        # set stacks and recurse
        self.current_node = self.parse_stack_w[-1]
        self.parse_stack_r.append(node)
        for chd in node:
            self.walk_markdown_ast(chd)
        self.parse_stack_r.pop()
        assert r_depth == len(self.parse_stack_r)
        # restore previous write stack
        self.parse_stack_w = self.parse_stack_w[:w_depth]
        self.current_node = self.parse_stack_w[-1]
        assert w_depth == len(self.parse_stack_w)

        self.dispatch(False, n, node, res)
        # add text
        if node.tail and node.tail.strip():
            self.append_text(node.tail)

    def dispatch(self, entering, n, node, *args):
        fn_prefix = "visit" if entering else "depart"
        fn_name = "{0}_{1}".format(fn_prefix, n)

        def x(*args): return self.dispatch_default(entering, *args)
        # if entering:
        # print(" " * len(self.parse_stack_r) * 2, node.tag, node.text[:40] if
        # node.text else "")
        return getattr(self, fn_name, x)(node, *args)

    def dispatch_default(self, entering, node, *args):
        if entering:
            self.document.reporter.warning(
                "markdown node with unknown tag: %s" % node.tag, nodes.Text(node.text))

    def append_text(self, text):

        if not self.raw_html_k:
            text1 = text
        else:
            text1 = self.raw_html_k.sub(
                lambda m: self.raw_html[m.group(0)], text)

        strip_p = False

        if text1 == text:
            content = nodes.Text(text)

        # hacky workaround for fenced_code
        elif text1.startswith("<pre><code") and text1.endswith("</code></pre>"):
            text = text1[10:-13]
            if text.startswith(">"):
                text = html.unescape(text[1:])
                content = nodes.literal_block(text, text)
            elif text.startswith(' class="'):
                text = text[8:]
                langi = text.find('"')
                lang = text[:langi]
                text = text[langi + 2:].rstrip("\n")
                text = html.unescape(text)

                content = nodes.literal_block(text, text, language=lang)
            else:
                self.document.reporter.warning(
                    "aborting attempt to parse invalid raw code block", nodes.Text(text1))
                content = nodes.raw(text1, text1, format='html')
            strip_p = True

        else:
            tags = MAYBE_HTML_TAG.findall(text1)
            # hacky heuristic to determine whether to strip <p> or not
            if not all((t in TAGS_INLINE for t in tags)):
                strip_p = True
            content = nodes.raw(text1, text1, format='html')

        parent = self.parse_stack_w[-1]
        if strip_p and len(parent) == 0 and isinstance(parent, nodes.paragraph):
            x = self.pop_node()
        self.parse_stack_w[-1] += content

    def reset_w_old(self):
        # reset parse_stack_w_old so that walk_markdown_ast rewinds there
        self.parse_stack_w_old = len(self.parse_stack_w)

    def append_node(self, node):
        self.parse_stack_w[-1] += node
        self.parse_stack_w.append(node)
        return node

    def pop_node(self):
        x = self.parse_stack_w.pop()
        self.reset_w_old()
        y = x.parent.children.pop()
        assert y is x
        return x

    def new_section(self, heading):
        section = nodes.section()
        anchor = to_html_anchor("".join(heading.itertext()))
        section['ids'] = [anchor]
        section['names'] = [anchor]
        return section

    def start_new_section(self, lvl, heading):
        while lvl <= self.parse_stack_h[-1]:
            x = self.parse_stack_w.pop()
            assert isinstance(x, nodes.section)
            self.parse_stack_h.pop()
        self.append_node(self.new_section(heading))
        self.reset_w_old()
        self.parse_stack_h.append(lvl)
        assert isinstance(self.parse_stack_w[-1], nodes.section)
        return nodes.title()

    def get_node_raw_html(self, node):
        htmlText = etree.tostring(node, encoding="unicode")
        try:
            node.text = ''
        except:
            pass
        try:
            node.tail = ''
        except:
            pass
        rawNode = nodes.raw(
            '', htmlText, format='html')
        return rawNode

    def visit_script(self, node):
        if node.attrib.get("type", "").split(";")[0] == "math/tex":
            node.attrib.pop("type")
            parent = self.parse_stack_r[-1]
            if parent.tag == "span":
                return nodes.math()
            elif parent.tag == "div":
                # sphinx mathjax crashes without these attributes present
                math = nodes.math_block()
                math["nowrap"] = None
                math["number"] = None
                return math
            else:
                self.document.reporter.warning(
                    "math/tex script with unknown parent: %s" % parent.tag)
        else:
            return IGNORE_ALL_CHILDREN

    def visit_input(self, node):
        return self.get_node_raw_html(node)

    def visit_ins(self, node):
        return self.get_node_raw_html(node)

    def visit_del(self, node):
        return self.get_node_raw_html(node)

    def visit_caption(self, node):
        return self.get_node_raw_html(node)

    def visit_mark(self, node):
        return self.get_node_raw_html(node)

    def visit_sub(self, node):
        return self.get_node_raw_html(node)

    def visit_p(self, node):
        return nodes.paragraph()

    def visit_span(self, node):
        if "MathJax_Preview" in node.attrib.get("class", "").split():
            node.attrib.pop("class")
            return IGNORE_ALL_CHILDREN
        return None

    def visit_div(self, node):
        if len(self.parse_stack_w) == 1:
            # top-level, ignore
            return None
        if "MathJax_Preview" in node.attrib.get("class", "").split():
            node.attrib.pop("class")
            return IGNORE_ALL_CHILDREN
        return None

    def visit_h1(self, node):
        return self.start_new_section(1, node)

    def visit_h2(self, node):
        return self.start_new_section(2, node)

    def visit_h3(self, node):
        return self.start_new_section(3, node)

    def visit_h4(self, node):
        return self.start_new_section(4, node)

    def visit_h5(self, node):
        return self.start_new_section(5, node)

    def visit_h6(self, node):
        return self.start_new_section(6, node)

    def visit_strong(self, node):
        return nodes.strong()

    def visit_em(self, node):
        return nodes.emphasis()

    def visit_br(self, node):
        return nodes.Text('\n')

    def visit_a(self, node):
        reference = nodes.reference()
        ids = node.attrib.get("id", "")
        if ids:
            reference['ids'] = [node.attrib.pop("id")]
        classes = node.attrib.get("classes", "")
        if classes:
            reference["classes"] = [node.attrib.pop("class")]
        href = node.attrib.pop('href', '')
        if href.endswith(".md"):
            href = href[:-3] + ".html"
        if href.startswith("/"):
            href = href[1:]
        # if not href.endswith(".html"):
        #     if len(href.split(".")[-1]) > 3:
        #         href = href + ".html"
        reference['refuri'] = href

        return reference

    def visit_ol(self, node):
        ol = nodes.enumerated_list()
        ids = node.attrib.get("id", "")
        if ids:
            ol['ids'] = [node.attrib.pop("id")]
        classes = node.attrib.get("classes", "")
        if classes:
            ol["classes"] = [node.attrib.pop("class")]
        return ol

    def visit_ul(self, node):
        ul = nodes.bullet_list()
        ids = node.attrib.get("id", "")
        if ids:
            ul['ids'] = [node.attrib.pop("id")]
        classes = node.attrib.get("classes", "")
        if classes:
            ul["classes"] = [node.attrib.pop("class")]
        return ul

    def visit_li(self, node):
        thisItem = nodes.list_item()
        ids = node.attrib.get("id", "")
        if ids:
            thisItem['ids'] = [node.attrib.pop("id")]
        classes = node.attrib.get("classes", "")
        if classes:
            thisItem["classes"] = [node.attrib.pop("class")]
        ch = list(node)
        # extra "paragraph" is needed to avoid breaking docutils assumptions
        if not ch or ch[0].tag in TAGS_INLINE:
            self.append_node(thisItem)
            return nodes.paragraph()
        else:
            return thisItem

    def visit_img(self, node):
        image = nodes.image()
        image['uri'] = node.attrib.pop('src', '')
        # image['uri'] = "/fixed/image.png"
        alt = node.attrib.pop('alt', '')
        if alt:
            image += nodes.Text(alt)
        return image

    def visit_hr(self, node):
        return nodes.transition()

    def visit_blockquote(self, node):
        return nodes.block_quote()

    def visit_iframe(self, node):
        if not node.text:
            node.text = html.unescape("placeholder")
        return self.get_node_raw_html(node)

    def visit_code(self, node):
        parent = self.parse_stack_r[-1]
        if node.text:
            node.text = html.unescape(node.text)

        if len(parent) == 1 and parent.tag in ("p", "pre") and not parent.text:

            x = self.pop_node()
            assert isinstance(x, nodes.paragraph) or isinstance(
                x, nodes.literal_block)
            block = nodes.literal_block()
            # note: this isn't yet activated because fenced_code extension
            # outputs raw html block, not a regular markdown ast tree. instead
            # what is actually run is the hacky workaround in append_text
            lang = node.attrib.get("class", "").replace("language-", "")
            if lang:
                node.attrib.pop("class")
                block["language"] = lang
            else:
                block["language"] = "text"
            return block
        else:
            return nodes.literal()

    def visit_table(self, node):
        # docutils html writer crashes without tgroup/colspec
        table = nodes.table()
        table['classes'] = ["colwidths-auto"]
        self.append_node(table)
        tgroup = nodes.tgroup()
        maxrow = max(len(row.findall("td")) for row in node.findall("*/tr"))
        for _ in range(maxrow):
            tgroup += nodes.colspec()
        tgroup['stub'] = None
        return tgroup

    def visit_thead(self, node):
        return nodes.thead()

    def visit_tbody(self, node):
        return nodes.tbody()

    def visit_tr(self, node):
        return nodes.row()

    def visit_th(self, node):
        return nodes.entry()

    def visit_td(self, node):
        return nodes.entry()

    def visit_pre(self, node):
        if node.text:
            node.text = html.unescape(node.text)
        return nodes.literal_block()

    def visit_abbr(self, node):
        htmlText = html.unescape(node.text)
        title = node.attrib.pop("title")
        node.text = ''
        htmlText = """<abbr title = '%(title)s'>%(htmlText)s</abbr>""" % locals()

        abbr = nodes.raw(
            '', htmlText, format='html')

        return abbr

    def visit_dl(self, node):
        return nodes.definition_list()

    def visit_dt(self, node):
        if node.text:
            node.text = html.unescape(node.text)
        return nodes.term()

    def visit_dd(self, node):
        if node.text:
            node.text = html.unescape(node.text)
        return nodes.definition()

    def visit_sup(self, node):
        sup = nodes.superscript()
        ids = node.attrib.get("id", "")
        if ids:
            sup['ids'] = [node.attrib.pop("id")]
        return sup
