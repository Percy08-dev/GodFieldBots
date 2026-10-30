"""Generate identical owned/borrowed typed-journal-experiment-v1 concrete DTOs.

Source: Bots journal.py/domain.py/validation.py and pinned Simulator DTO contract.
No JSON subtree is arbitrary in this Journal contract: no Value fields are used.
Regenerate: python benchmarks/native_preprocessing/generate_schema.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'benchmarks/native_preprocessing/rust/src'
BORROWED_OUT = Path(__file__).resolve().parent / 'borrowed.rs'
# suffix ? = required nullable, * = optional non-null, ~ = optional nullable.
STRUCTS = {
    'Selection': {'index': 'u64', 'policy_version': 'Text', 'logprob': 'f64?', 'value': 'f64?'},
    'Winner': {'winner': 'Player'},
    'CatalogMeta': {'catalog_schema_version': 'u32', 'catalog_version': 'Text',
                    'rule_set_version': 'Text', 'locale': 'Text', 'content_sha256': 'Text',
                    'definition_count': 'u32'},
    'Description': {'id': 'Text', 'name': 'Text', 'description': 'Text'},
    'Definition': {'definition_id': 'Text', 'display_name': 'Text', 'kind': 'Text',
                   'attack': 'i32*', 'defense': 'i32*', 'ability_value': 'i32*',
                   'attribute': 'Text*', 'price': 'u32*', 'mp_cost': 'u32*',
                   'additional_attack': 'bool*', 'hit_rate': 'u32*', 'ability': 'Text*',
                   'effect_text': 'Text*', 'curse': 'Description*', 'guardian': 'Description*'},
    'Status': {'disease': 'Text?', 'fog': 'bool', 'flash': 'bool', 'dream': 'bool', 'dark_cloud': 'bool'},
    'Artifact': {'instance_id': 'u64', 'definition_id': 'Text'},
    'Own': {'player_id': 'Player', 'hp': 'i32', 'mp': 'u32', 'money': 'u32',
            'hand': 'Vec<Artifact>', 'learned_miracles': 'Vec<Text>', 'status': 'Status',
            'guardian': 'Text?', 'ascended': 'bool'},
    'Opponent': {'player_id': 'Player', 'hp': 'i32*', 'mp': 'u32*', 'money': 'u32*',
                 'learned_miracles': 'Vec<Text>', 'status': 'Status*', 'guardian': 'Text~',
                 'ascended': 'bool'},
    'Observation': {'observation_schema_version': 'Text', 'action_schema_version': 'u32',
                    'actionable': 'bool', 'typed_action_schema': 'Vec<Schema>', 'state_revision': 'u64',
                    'viewer_player_id': 'Player', 'turn': 'u32', 'turns_until_end_of_world': 'u32',
                    'first_player': 'Player', 'active_player': 'Player', 'current_player': 'Player?',
                    'phase': 'Text', 'required_choice': 'Text?', 'end_times': 'bool',
                    'own': 'Own', 'opponent': 'Opponent', 'result': 'Outcome?'},
    'AtomicBody': {'required_choice': 'Text?', 'usable_hand_elements': 'Vec<u64>',
                   'usable_learned_miracles': 'Vec<Text>', 'exchange_artifacts': 'Vec<u64>',
                   'sell_artifacts': 'Vec<u64>', 'buy_artifacts': 'Vec<u64>',
                   'valid_targets': 'Vec<Player>', 'can_pray': 'bool', 'can_discard': 'bool',
                   'can_sacrifice': 'bool', 'can_allow': 'bool'},
    'AtomicOptions': {'viewer_player_id': 'Player', 'state_revision': 'u64', 'action_options': 'AtomicBody'},
    'Input': {'observation': 'Observation', 'events': 'Vec<Event>', 'history_end': 'Text?',
              'typed_metadata': 'Vec<Schema>', 'catalog': 'BTreeMap<Text, Definition>',
              'action_options': 'AtomicOptions?'},
    'Candidates': {'actions': 'Vec<Action>', 'generator_version': 'Text', 'scorer_version': 'Text',
                   'config_hash': 'Text', 'attempts': 'u64', 'public_prunes': 'u64',
                   'approximate_drops': 'u64', 'latency_ms': 'f64', 'cutoff': 'bool',
                   'cutoff_reasons': 'Vec<Text>'},
    'Command': {'action': 'Action', 'command_id': 'Text', 'expected_revision': 'u64'},
    'Acknowledgement': {'command_id': 'Text', 'previous_revision': 'u64', 'state_revision': 'u64',
                        'status': 'Text', 'result': 'Outcome?'},
    'Element': {'source': 'Text', 'instance_id': 'u64*', 'definition_id': 'Text*'},
    'Action': {'type': 'Text', 'ordered_elements': 'Vec<Element>*', 'target': 'Player~',
               'instance_ids': 'Vec<u64>*', 'artifact': 'u64*', 'hp': 'i32*', 'mp': 'u32*',
               'money': 'u32*', 'sold_artifact': 'u64*', 'ordered_defenses': 'Vec<Defense>*',
               'accept': 'bool*'},
    'Event': {'type': 'Text', 'turn': 'u32*', 'active_player': 'Player*', 'player': 'Player*',
              'definition_id': 'Text*', 'mp_spent': 'u32*', 'attacker': 'Player*', 'target': 'Player*',
              'attack': 'i32*', 'amount': 'i32*', 'hp_delta': 'i32*', 'mp_delta': 'i32*',
              'money_delta': 'i32*', 'buyer': 'Player*', 'seller': 'Player*', 'price': 'u32*',
              'from': 'Player*', 'to': 'Player*', 'status': 'Text*', 'active': 'bool*',
              'result': 'Outcome~', 'remaining_hp': 'i32*', 'hp': 'i32*'},
    # Concrete JSON schema descriptors; their scalar domains match action_descriptors.
    'Schema': {'type': 'Text*', 'minimum': 'i128*', 'maximum': 'i128*',
               'sources': 'Vec<Text>*', 'enum': 'Vec<Player>*', 'one_of': 'Vec<Schema>*',
               'required_fields': 'BTreeMap<Text, Schema>*', 'optional_fields': 'BTreeMap<Text, Schema>*',
               'additional_fields': 'bool*', 'const': 'Text*', 'items': 'Box<Schema>*', 'ordered': 'bool*'},
    'Record': {'kind': 'Text', 'schema_version': 'u32*', 'feature_version': 'Text*',
               'reward_version': 'Text*', 'policy_input_version': 'Text*', 'use_action_options': 'bool*',
               'end_projection_version': 'Text*', 'catalog': 'CatalogMeta*', 'episode_id': 'Text*',
               'viewer': 'Player*', 'policy': 'Text*', 'run_id': 'Text*', 'game_id': 'Text*',
               'action_config_hash': 'Text*', 'generator_version': 'Text*', 'scorer_version': 'Text*',
               'decision': 'u64*', 'input': 'Input*', 'candidates': 'Candidates*',
               'selection': 'Selection*', 'command': 'Command*', 'selection_elapsed_ms': 'f64*',
               'response': 'Acknowledgement*', 'next_decision': 'u64~', 'reward': 'i32*',
               'terminal': 'bool*', 'end_kind': 'Text*', 'outcome': 'Outcome~', 'terminal_reward': 'i32~',
               'final_observation': 'Observation~', 'pending_decisions': 'Vec<u64>*',
               'simulator_abort_reason': 'Text~'},
}

SHAPES = {
    'Element': ('source', {'hand_artifact': ('instance_id', ''),
                            'learned_miracle': ('definition_id', '')}),
    'Action': ('type', {'pray': ('', ''), 'play_sequence': ('ordered_elements', 'target'),
                        'discard': ('instance_ids', ''), 'sacrifice': ('instance_ids', ''),
                        'exchange': ('artifact hp mp money', ''),
                        'sell': ('artifact sold_artifact target', ''), 'buy': ('artifact target', ''),
                        'respond': ('ordered_defenses', ''), 'respond_sequence': ('ordered_defenses', ''),
                        'buy_decision': ('accept', '')}),
    'Event': ('type', {
        'turn_started': ('turn active_player', ''), 'action_accepted': ('player', ''),
        **{tag: ('player definition_id', '') for tag in ('artifact_used', 'artifact_drawn',
             'devil_appeared', 'artifact_discarded', 'guardian_set', 'guardian_left', 'guardian_acted')},
        'miracle_used': ('player definition_id mp_spent', ''),
        'attack_declared': ('attacker target attack', ''), 'attack_missed': ('attacker target', ''),
        'damage_applied': ('player amount', 'remaining_hp'), 'healed': ('player amount', 'hp'),
        'resource_changed': ('player hp_delta mp_delta money_delta', ''),
        'buy_offered': ('buyer seller definition_id price', ''),
        'sale_offered': ('seller target definition_id price', ''),
        'artifact_transferred': ('from to definition_id price', ''),
        'sun_amulet_activated': ('player', ''), 'status_changed': ('player status active', ''),
        'phenomenon_invoked': ('definition_id', ''), 'player_ascended': ('player', ''),
        'player_defeated': ('player', ''), 'game_ended': ('result', '')}),
    'Record': ('kind', {
        'header': (('schema_version feature_version reward_version policy_input_version use_action_options '
                   'end_projection_version catalog episode_id viewer policy run_id game_id action_config_hash '
                   'generator_version scorer_version'), ''),
        'decision': ('decision input candidates selection command selection_elapsed_ms', ''),
        'acknowledgement': ('decision response', ''),
        'transition': ('decision next_decision reward terminal', ''),
        'end': ('end_kind outcome terminal_reward final_observation pending_decisions simulator_abort_reason', '')}),
}

EXTRA = {
    'Winner': '',
    'Definition': '''one_of(self.kind.as_ref(), &["weapons","armor","miracles","sundries","trade","guardians","phenomena","devils"])?;
        if let Presence::Value(v) = &self.attribute { one_of(v.as_ref(), &["fire","water","wood","stone","light","darkness"])?; }
        if let Presence::Value(v) = self.hit_rate { ensure(v <= 100, "hit rate range")?; }''',
    'Status': '''if let Some(v) = &self.disease { one_of(v.as_ref(), &["cold","fever","hell","heaven"])?; }''',
    'Observation': '''ensure(self.observation_schema_version.as_ref() == "1" && self.action_schema_version == 1, "observation schema")?;
        one_of(self.phase.as_ref(), &["awaiting_action","awaiting_response","resolving","terminal"])?;
        if let Some(v) = &self.required_choice { one_of(v.as_ref(), &["turn_action","defense","buy_accept_reject","additional_choice"])?; }
        ensure(self.own.player_id == self.viewer_player_id && self.opponent.player_id.0 == 3-self.viewer_player_id.0, "participant identity")?;''',
    'Acknowledgement': '''one_of(self.status.as_ref(), &["active","terminal","aborted"])?;''',
    'Action': '''if let Presence::Value(defenses) = &self.ordered_defenses {
            for defense in defenses { ensure(matches!((self.r#type.as_ref(), defense),
                ("respond", Defense::Instance(_)) | ("respond_sequence", Defense::Element(_))), "ordered defense tag/shape")?; }
        }
        if matches!(self.r#type.as_ref(), "buy"|"sell") { ensure(matches!(self.target, Presence::Value(_)), "target required non-null")?; }''',
    'Schema': '''let required: &[&str] = if self.one_of.present() { &["one_of"] } else {
            match self.r#type.value().map(AsRef::<str>::as_ref) {
                Some("integer") if self.r#enum.present() => &["type","enum","sources"],
                Some("integer") if self.sources.present() => &["type","minimum","maximum","sources"],
                Some("integer") => &["type","minimum","maximum"],
                Some("string") if self.r#const.present() => &["type","const"],
                Some("string") => &["type","sources"],
                Some("object") => &["type","required_fields","optional_fields","additional_fields"],
                Some("pray"|"play_sequence"|"discard"|"sacrifice"|"exchange"|"sell"|"buy"|"respond"|"respond_sequence"|"buy_decision") => &["type","required_fields","optional_fields","additional_fields"],
                Some("array") => &["type","items","ordered"],
                Some("boolean"|"null") => &["type"],
                _ => return Err("unknown typed metadata schema".into()),
            }
        };
        shape(&self.presence(), required, &[])?;''',
    'Record': '''if self.kind.as_ref() == "header" { ensure(matches!(self.schema_version, Presence::Value(4)), "journal schema")?; }
        if self.kind.as_ref() == "end" { one_of(self.end_kind.value().unwrap().as_ref(), &["natural","truncated","aborted","cancelled","incomplete"])?; }''',
}


def generate(borrowed: bool) -> str:
    names = set(STRUCTS) | {'Outcome', 'Defense'}
    lifetime_names = names - {'Winner'}
    # Winner has no string-containing descendants; every other struct does.

    def rust_type(raw: str) -> str:
        import re
        if not borrowed:
            return raw.replace('Text', 'String')
        for name in sorted(lifetime_names | {'Text'}, key=len, reverse=True):
            raw = re.sub(r'\b' + name + r'\b', name + "<'a>", raw)
        return raw

    lines = ['// @generated by generate_schema.py; edit its schema, then regenerate.',
             'use serde::{Deserialize, Serialize};',
             'use crate::contract::*;',
             *(['use crate::feasibility::Text;'] if borrowed else []),
             '#[derive(Debug, Serialize)]',
             '#[serde(rename_all="snake_case")]',
             "pub enum Outcome { Draw, Win(Winner) }",
             '''impl<'de> Deserialize<'de> for Outcome {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self,D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Outcome;
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {
                f.write_str("string draw or single-key win object")
            }
            fn visit_str<E:serde::de::Error>(self,value:&str)->Result<Outcome,E> {
                if value=="draw" {Ok(Outcome::Draw)} else {Err(E::custom("invalid outcome string"))}
            }
            fn visit_map<M:serde::de::MapAccess<'de>>(self,mut map:M)->Result<Outcome,M::Error> {
                let key=map.next_key::<String>()?.ok_or_else(||serde::de::Error::custom("missing win outcome"))?;
                if key!="win" {return Err(serde::de::Error::custom("only win outcome object is allowed"));}
                let winner=map.next_value::<Winner>()?;
                if map.next_key::<String>()?.is_some(){return Err(serde::de::Error::custom("duplicate or extra outcome key"));}
                Ok(Outcome::Win(winner))
            }
        }
        decoder.deserialize_any(Visitor)
    }
}''',
             'impl Validate for Outcome { fn validate(&self)->Result<(),String> { match self { Self::Draw => Ok(()), Self::Win(w) => w.validate() } } }']
    # Outcome has no lifetime regardless of owned/borrowed text.
    lifetime_names.remove('Outcome')
    for name, fields in STRUCTS.items():
        lt = "<'a>" if borrowed and name in lifetime_names else ''
        lines += ['#[derive(Debug, Deserialize, Serialize)]', '#[serde(deny_unknown_fields)]',
                  f'pub struct {name}{lt} {{']
        for field, raw in fields.items():
            mode = raw[-1] if raw[-1] in '?*~' else ''
            typ = rust_type(raw.rstrip('?*~'))
            attrs = []
            if mode == '?':
                typ = f'Option<{typ}>'
                attrs.append('deserialize_with="crate::feasibility::required_nullable"')
            elif mode in ('*', '~'):
                typ = f'Presence<{typ}>'
                attrs += ['default', 'skip_serializing_if="Presence::missing"']
            if borrowed and "<'a>" in typ:
                attrs.append('borrow')
            if attrs:
                lines.append('    #[serde(' + ', '.join(attrs) + ')]')
            lines.append(f'    pub r#{field}: {typ},')
        lines.append('}')
        lines += [f'impl{lt} {name}{lt} {{', '    fn presence(&self) -> [(&\'static str, bool); '
                  + str(len(fields)) + '] { [']
        for field, raw in fields.items():
            present = f'self.r#{field}.present()' if raw[-1] in '*~' else 'true'
            lines.append(f'        ("{field}", {present}),')
        lines += ['    ] }', '}', f'impl{lt} Validate for {name}{lt} {{',
                  '    fn validate(&self) -> Result<(), String> {']
        for field, raw in fields.items():
            lines.append(f'        self.r#{field}.validate()?;')
            if raw.endswith('*'):
                lines.append(f'        ensure(!matches!(self.r#{field}, Presence::Null), "non-null {name}.{field}")?;')
        if name in SHAPES:
            tag, shapes = SHAPES[name]
            lines.append(f'        let (required, optional): (&[&str], &[&str]) = match self.r#{tag}.as_ref() {{')
            for variant, (required, optional) in shapes.items():
                req = ','.join('"' + f + '"' for f in [tag, *required.split()])
                opt = ','.join('"' + f + '"' for f in optional.split())
                lines.append(f'            "{variant}" => (&[{req}], &[{opt}]),')
            lines += ['            _ => return Err("unknown scalar tag".into()),', '        };',
                      '        shape(&self.presence(), required, optional)?;']
        if name in EXTRA:
            lines.append(EXTRA[name])
        lines += ['        Ok(())', '    }', '}']
    lt = "<'a>" if borrowed else ''
    de = "<'de:'a,'a>" if borrowed else "<'de>"
    result_lt = "<'de>" if borrowed else ''
    lines.append(f'''
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Defense{lt} {{ Instance(u64), Element(Element{lt}) }}
impl{de} Deserialize<'de> for Defense{lt} {{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self,D::Error> {{
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {{
            type Value = Defense{result_lt};
            fn expecting(&self,f:&mut std::fmt::Formatter)->std::fmt::Result {{ f.write_str("integer or element") }}
            fn visit_u64<E:serde::de::Error>(self,v:u64)->Result<Self::Value,E> {{ Ok(Defense::Instance(v)) }}
            fn visit_map<M:serde::de::MapAccess<'de>>(self,m:M)->Result<Self::Value,M::Error> {{
                Element::deserialize(serde::de::value::MapAccessDeserializer::new(m)).map(Defense::Element)
            }}
        }}
        d.deserialize_any(V)
    }}
}}
impl{lt} Validate for Defense{lt} {{ fn validate(&self)->Result<(),String> {{
    match self {{ Self::Instance(_) => Ok(()), Self::Element(v) => v.validate() }}
}} }}
''')
    source = ('\n'.join(lines) + '\n').replace('BTreeMap<', 'UniqueMap<').replace(
        'self.observation_schema_version.as_ref() ==', 'text(&self.observation_schema_version) ==').replace(
        'self.kind.as_ref() ==', 'text(&self.kind) ==')
    if borrowed:
        source = source.replace('use crate::contract::*;', 'use super::contract::*;').replace(
            'use crate::feasibility::Text;', 'use super::text::Text;').replace(
            'crate::feasibility::required_nullable', 'super::text::required_nullable')
    source = source.replace('derive(Debug, Deserialize, Serialize)', 'derive(Debug, PartialEq, Deserialize, Serialize)').replace(
        'derive(Debug, Serialize)', 'derive(Debug, PartialEq, Serialize)')
    return subprocess.run(['rustfmt', '--edition', '2024', '--emit', 'stdout'],
                          input=source, text=True, capture_output=True, check=True).stdout


if __name__ == '__main__':
    for borrowed in (False, True):
        (BORROWED_OUT if borrowed else OUT / 'owned.rs').write_text(generate(borrowed))
