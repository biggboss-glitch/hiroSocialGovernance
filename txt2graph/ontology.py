"""
自定义实体类型定义
只提取现实生活中真实存在的、可以有行动的实体
"""

from pydantic import Field
from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel


# ============== 实体类型定义 ==============

class Person(EntityModel):
    """A real, named individual. Must have a specific name like "马化腾" or "Elon Musk". NOT generic terms like "某人", "用户", pronouns, or abstract roles."""
    role: EntityText = Field(
        description="职业或职位",
        default=None
    )
    affiliation: EntityText = Field(
        description="所属组织",
        default=None
    )


class Organization(EntityModel):
    """A real organization with an official name like "武汉大学", "联合国". NOT generic terms like "大学", "政府", "组织"."""
    org_type: EntityText = Field(
        description="组织类型",
        default=None
    )
    location: EntityText = Field(
        description="所在地",
        default=None
    )


class Company(EntityModel):
    """A real company with registered name like "腾讯", "Apple Inc.". NOT generic terms like "科技公司", "某企业"."""
    industry: EntityText = Field(
        description="所属行业",
        default=None
    )
    headquarters: EntityText = Field(
        description="总部位置",
        default=None
    )


class Location(EntityModel):
    """A real geographic location like "北京", "硅谷", "故宫". NOT generic terms like "某地", "一个城市"."""
    location_type: EntityText = Field(
        description="地点类型",
        default=None
    )
    country: EntityText = Field(
        description="所属国家",
        default=None
    )


class Product(EntityModel):
    """A real product/service with brand name like "iPhone", "微信". NOT generic categories like "手机", "软件"."""
    category: EntityText = Field(
        description="产品类别",
        default=None
    )
    manufacturer: EntityText = Field(
        description="制造商",
        default=None
    )


class Event(EntityModel):
    """A real, documented event like "2024巴黎奥运会", "新冠疫情". NOT generic terms like "会议", "活动"."""
    event_type: EntityText = Field(
        description="事件类型",
        default=None
    )
    date: EntityText = Field(
        description="发生日期",
        default=None
    )


class Media(EntityModel):
    """A real media outlet like "人民日报", "CNN", "微博". NOT generic terms like "媒体", "新闻"."""
    media_type: EntityText = Field(
        description="媒体类型",
        default=None
    )


# ============== 边类型定义 ==============

class WorksFor(EdgeModel):
    """Employment or affiliation relationship."""
    position: EntityText = Field(
        description="职位",
        default=None
    )


class LocatedIn(EdgeModel):
    """Geographic location relationship."""
    pass


class PartOf(EdgeModel):
    """Part-of or subsidiary relationship."""
    pass


class Produces(EdgeModel):
    """Production or creation relationship."""
    pass


class ParticipatesIn(EdgeModel):
    """Participation in an event."""
    role: EntityText = Field(
        description="参与角色",
        default=None
    )


class Collaborates(EdgeModel):
    """Collaboration or partnership relationship."""
    pass


class Competes(EdgeModel):
    """Competitive relationship."""
    pass


class Reports(EdgeModel):
    """Media reporting or coverage relationship."""
    pass


# ============== 本体配置 ==============

# 实体类型字典
ENTITY_TYPES = {
    "Person": Person,
    "Organization": Organization,
    "Company": Company,
    "Location": Location,
    "Product": Product,
    "Event": Event,
    "Media": Media,
}

# 边类型字典
EDGE_TYPES = {
    "WORKS_FOR": WorksFor,
    "LOCATED_IN": LocatedIn,
    "PART_OF": PartOf,
    "PRODUCES": Produces,
    "PARTICIPATES_IN": ParticipatesIn,
    "COLLABORATES": Collaborates,
    "COMPETES": Competes,
    "REPORTS": Reports,
}
