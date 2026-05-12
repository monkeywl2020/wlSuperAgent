# -*- coding: utf-8 -*-
"""
问答提取模块 (Q&A Extractor)

从记忆文本中提取问答对
"""

# 导入标准库
from typing import List, Dict, Any, Optional
import re
import time

# 导入第三方库
from loguru import logger

# 导入LLM接口
from ..llm import LLMProvider


class QAPair:
    """问答对"""

    def __init__(self, question: str, answer: str, source: str = ""):
        """
        初始化问答对

        Args:
            question: 问题
            answer: 回答
            source: 来源（如日期）
        """
        self.question = question.strip()
        self.answer = answer.strip()
        self.source = source

    def to_text(self) -> str:
        """转换为文本（Q:问题\nA:回答 格式）"""
        return f"Q: {self.question}\nA: {self.answer}"

    def to_dict(self) -> Dict[str, str]:
        """转换为字典"""
        return {
            "question": self.question,
            "answer": self.answer,
            "source": self.source
        }

    def get_length(self) -> int:
        """获取问答对文本长度"""
        return len(self.to_text())

    def __repr__(self):
        return f"QAPair(question='{self.question[:30]}...', answer='{self.answer[:30]}...')"


class QAExtractor:
    """问答提取器

    从记忆文本中提取问答对，输出格式为：
    Q:问题
    A:回答
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_template: str = ""
    ):
        """
        初始化问答提取器

        Args:
            llm_provider: LLM接口，用于提取问答对
            prompt_template: QA提取提示词模板
        """
        self.llm_provider = llm_provider

        if prompt_template:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = """请从以下记忆文本中提取问答对。
要求：
1. 每个问答对应该是有意义的信息（重要事实、偏好、约定等）
2. 问题要简洁，能概括回答的核心
3. 回答要完整，保留关键信息
4. 如果文本中没有明确的信息，跳过
5. 输出的格式必须是：Q:问题\nA:回答（每对之间用空行分隔）

记忆文本：
{memory_text}

请输出问答对（只输出问答对，不要其他内容）："""

    def extract(self, memory_text: str, source: str = "") -> List[QAPair]:
        """
        从记忆文本中提取问答对

        Args:
            memory_text: 记忆文本
            source: 来源标识（如日期）

        Returns:
            问答对列表
        """
        if not memory_text or not memory_text.strip():
            return []

        # 如果没有 LLM provider，使用正则表达式提取
        if self.llm_provider is None:
            return self._extract_by_regex(memory_text, source)

        try:
            return self._extract_by_llm(memory_text, source)
        except Exception as e:
            logger.warning(f"LLM QA extraction failed, falling back to regex: {e}")
            return self._extract_by_regex(memory_text, source)

    def _extract_by_llm(self, memory_text: str, source: str) -> List[QAPair]:
        """使用 LLM 提取问答对"""
        wlstartTime1 = time.perf_counter()
        logger.info(f"QAExtractor::_extract_by_llm start, memory_text: {memory_text[:100]}...")

        prompt = self._prompt_template.format(memory_text=memory_text)
        response = self.llm_provider.chat([{"role": "user", "content": prompt}])

        wlstartTime2 = time.perf_counter()
        timeCost1 = wlstartTime2 - wlstartTime1
        logger.info(f"QAExtractor::_extract_by_llm cost1 time: {timeCost1:.4f} seconds")

        # 解析响应
        text = response if isinstance(response, str) else str(response)

        # 检查是否返回"无"（表示没有有用信息）
        if text.strip() == "无" or "无" in text[:10]:
            logger.info("QAExtractor::No useful information extracted")
            return []

        return self._parse_qa_text(text, source)

    def _extract_by_regex(self, memory_text: str, source: str) -> List[QAPair]:
        """使用正则表达式提取问答对（简单实现）"""
        # 尝试匹配 Q:... A:... 格式
        qa_pairs = self._parse_qa_text(memory_text, source)

        if qa_pairs:
            return qa_pairs

        # 如果没有匹配到QA格式，按行解析
        lines = memory_text.split('\n')
        current_section = ""
        qa_pairs = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题（## 开头的）
            if line.startswith('##'):
                current_section = line.lstrip('#').strip()
                continue

            # 如果是非标题内容，且有一定长度，尝试作为信息点
            if not line.startswith('#') and len(line) > 10:
                # 移除常见的 Markdown 符号
                line = re.sub(r'[*_`\[\]]', '', line)
                if line.startswith(('- ', '* ', '1. ')):
                    line = line.lstrip('-*1234567890. ')

                if len(line) >= 5:
                    # 生成问答
                    if current_section:
                        question = f"{current_section}相关：{line[:50]}"
                    else:
                        question = f"记忆：{line[:50]}"
                    qa_pairs.append(QAPair(question, line, source))

        return qa_pairs

    def _parse_qa_text(self, text: str, source: str) -> List[QAPair]:
        """解析QA格式文本"""
        qa_pairs = []

        # 分割QA对（支持 Q:xxx A:yyy 或 Q:xxx\nA:yyy 格式）
        pattern = r'Q:\s*(.+?)\s*A:\s*(.+?)(?=\n\s*Q:|\Z)'
        matches = re.findall(pattern, text, re.DOTALL)

        for question, answer in matches:
            question = question.strip()
            answer = answer.strip()
            if question and answer:
                qa_pairs.append(QAPair(question, answer, source))

        # 如果正则没匹配到，尝试简单的行解析
        if not qa_pairs:
            lines = text.split('\n')
            current_q = None
            current_a = []

            for line in lines:
                line = line.strip()
                if line.startswith('Q:') or line.startswith('Q：'):
                    if current_q and current_a:
                        qa_pairs.append(QAPair(current_q, '\n'.join(current_a), source))
                    current_q = line.lstrip('Q: Q：').strip()
                    current_a = []
                elif line.startswith('A:') or line.startswith('A：'):
                    current_a.append(line.lstrip('A: A：').strip())
                elif current_a:
                    current_a.append(line)

            if current_q and current_a:
                qa_pairs.append(QAPair(current_q, '\n'.join(current_a), source))

        return qa_pairs


class MemoryChunker:
    """记忆分块器

    两种分块模式：
    1. QA模式：每个QA对独立成块（不考虑chunk_size）
    2. 固定大小模式：按字符数分块
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50
    ):
        """
        初始化分块器

        Args:
            chunk_size: 块大小阈值（字符数），仅在非QA模式生效
            overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        将文本按chunk_size分块（带重叠）

        分块逻辑：
        - 分块边界按 chunk_size 划分：块1 [0, chunk_size]，块2 [chunk_size, 2*chunk_size]...
        - 存入 Embedding 时，每个块实际内容 = chunk + overlap（追加在后面）
        - 例如 chunk_size=100, overlap=20：
          - 块1: 0-100，实际存入 0-120
          - 块2: 100-200，实际存入 100-220
          - ...

        Args:
            text: 输入文本
            source: 来源标识

        Returns:
            块列表
        """
        if not text:
            return []

        text_length = len(text)

        # 如果文本小于chunk_size，直接返回
        if text_length <= self.chunk_size:
            return [{"text": text, "source": source, "index": 0}]

        # 分块
        chunks = []
        index = 0

        for chunk_start in range(0, text_length, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, text_length)

            # 追加 overlap 内容（在后面追加，不是两个chunk之间共享）
            overlap_end = min(chunk_end + self.overlap, text_length)

            # 获取主chunk内容
            main_text = text[chunk_start:chunk_end]
            # 获取overlap内容
            overlap_text = text[chunk_end:overlap_end]

            # 完整chunk内容 = main + overlap
            full_text = main_text + overlap_text

            if full_text.strip():
                chunks.append({
                    "text": full_text.strip(),
                    "source": source,
                    "index": index,
                    "main_text": main_text.strip(),  # 主分块内容（不含overlap）
                    "overlap_text": overlap_text.strip()  # overlap内容
                })
                index += 1

        return chunks if chunks else [{"text": text, "source": source, "index": 0}]

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """在范围内查找最佳分割点（换行符）"""
        # 先尝试在end附近找换行符
        for i in range(end, max(start, end - 100), -1):
            if i < len(text) and text[i] in '\n\r':
                return i + 1

        # 再尝试在整个范围内找换行符
        for i in range(end, start, -1):
            if i < len(text) and text[i] in '\n\r':
                return i + 1

        # 没找到换行符，直接在end处分割
        return end

    def create_qa_chunks(self, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """将QA对列表转换为chunk列表

        每个QA对独立成一个chunk，不受chunk_size限制

        Args:
            qa_pairs: QA对列表

        Returns:
            chunk列表
        """
        chunks = []
        for i, qa in enumerate(qa_pairs):
            chunks.append({
                "text": qa.to_text(),
                "source": qa.source,
                "index": i,
                "qa_pair": qa.to_dict()
            })
        return chunks
