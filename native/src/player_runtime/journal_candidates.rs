//! Independent Journal6 candidate validation over an original pending capture.
#[cfg(test)]
use super::shared_card_instance_contract;
use super::{
    ensure, error, game::action_from_parts, is_shared_card_instance_contract,
    search::Table as ActorTable,
};
use crate::structured_journal::{TARGETS, card_instance::KINDS};
use crate::{error::Result, json::ValueExt};
use serde::{
    Deserialize, Deserializer, Serialize,
    de::{MapAccess, Visitor, value::MapAccessDeserializer},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fmt, io};

#[derive(Deserialize)]
struct Capture<'a> {
    selected: usize,
    #[serde(borrow)]
    frame: Frame<'a>,
    #[serde(borrow)]
    generated: Generated<'a>,
}
#[derive(Deserialize)]
struct Generated<'a> {
    #[serde(borrow, deserialize_with = "table_object")]
    table: Table<'a>,
}
#[derive(Deserialize)]
struct Frame<'a> {
    actor: u64,
    frame_hash: &'a str,
    #[serde(borrow)]
    payload: FramePayload<'a>,
}
#[derive(Deserialize)]
struct FramePayload<'a> {
    #[serde(borrow)]
    observation: Observation<'a>,
    #[serde(borrow)]
    typed_metadata: Vec<ActionKind<'a>>,
}
#[derive(Deserialize)]
struct Observation<'a> {
    #[serde(borrow)]
    own: Own<'a>,
}
#[derive(Deserialize)]
struct Own<'a> {
    #[serde(borrow)]
    cards: Vec<Card<'a>>,
}
#[derive(Deserialize)]
struct Card<'a> {
    card_ref: &'a str,
}
#[derive(Deserialize)]
struct ActionKind<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table<'a> {
    action_contract: Value,
    frame_hash: &'a str,
    #[serde(borrow)]
    kinds: Vec<&'a str>,
    numerics: Vec<[i64; 4]>,
    offsets: Vec<usize>,
    presence: Vec<[bool; 4]>,
    purpose: &'a str,
    slots: Vec<u8>,
    synthetic_capacity: bool,
    table_hash: &'a str,
    #[serde(borrow)]
    targets: Vec<&'a str>,
}

fn table_object<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Table<'de>, D::Error> {
    // Derived structs also accept positional arrays. The writer wire format is
    // a closed JSON object, so always use the derived map-field validation.
    struct Object;
    impl<'de> Visitor<'de> for Object {
        type Value = Table<'de>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a candidate table object")
        }

        fn visit_map<M: MapAccess<'de>>(
            self,
            map: M,
        ) -> std::result::Result<Self::Value, M::Error> {
            Table::deserialize(MapAccessDeserializer::new(map))
        }
    }
    deserializer.deserialize_map(Object)
}

// Alphabetical field order matches the existing canonical JSON. The pinned
// contract is a sorted serde_json::Value; remaining validated values are ASCII
// codebook strings, nonnegative offsets, bounded integers and bools.
#[derive(Serialize)]
struct CanonicalTable<'a, Kind, Slot> {
    action_contract: &'a Value,
    frame_hash: &'a str,
    kinds: &'a [Kind],
    numerics: &'a [[i64; 4]],
    offsets: &'a [usize],
    presence: &'a [[bool; 4]],
    purpose: &'a str,
    slots: &'a [Slot],
    synthetic_capacity: bool,
    targets: &'a [Kind],
}
struct HashWriter(Sha256);
impl io::Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) struct Candidates<'a> {
    pub frame_hash: &'a str,
    pub table_hash: &'a str,
    pub selected: usize,
    pub count: usize,
    pub elements: usize,
    pub action: Vec<u8>,
    pub payload: Vec<u8>,
    pub sha256: String,
    #[cfg(test)]
    parsed_columns_bytes: usize,
    #[cfg(test)]
    parsed_frame_bytes: usize,
}

const fn align_i64(offset: usize) -> usize {
    (offset + 7) & !7
}

pub(super) fn check(
    whole: &[u8],
    max_positions: usize,
    output_bytes: usize,
) -> Result<Candidates<'_>> {
    let Capture {
        selected,
        frame,
        generated: Generated { table },
    } = serde_json::from_slice(whole)
        .map_err(|exc| error(format!("writer candidate JSON: {exc}")))?;
    let content = CanonicalTable {
        action_contract: &table.action_contract,
        frame_hash: table.frame_hash,
        kinds: &table.kinds,
        numerics: &table.numerics,
        offsets: &table.offsets,
        presence: &table.presence,
        purpose: table.purpose,
        slots: &table.slots,
        synthetic_capacity: table.synthetic_capacity,
        targets: &table.targets,
    };
    #[allow(unused_mut)]
    let mut checked = check_columns(
        selected,
        frame,
        content,
        table.table_hash,
        max_positions,
        output_bytes,
    )?;
    #[cfg(test)]
    {
        checked.parsed_columns_bytes = vec_bytes(&table.kinds)
            + vec_bytes(&table.targets)
            + vec_bytes(&table.numerics)
            + vec_bytes(&table.presence)
            + vec_bytes(&table.offsets)
            + vec_bytes(&table.slots);
    }
    Ok(checked)
}

pub(super) fn check_actor_table<'a>(
    frame: &'a Value,
    selected: usize,
    table: &'a ActorTable,
    max_positions: usize,
    output_bytes: usize,
) -> Result<Candidates<'a>> {
    let payload = frame.require("payload")?;
    let cards = payload
        .require("observation")?
        .require("own")?
        .require("cards")?
        .array()?;
    let metadata = payload.require("typed_metadata")?.array()?;
    let frame = Frame {
        actor: frame.require("actor")?.integer(1, 2)? as u64,
        frame_hash: frame.require("frame_hash")?.string()?,
        payload: FramePayload {
            observation: Observation {
                own: Own {
                    cards: cards
                        .iter()
                        .map(|card| {
                            Ok(Card {
                                card_ref: card.require("card_ref")?.string()?,
                            })
                        })
                        .collect::<Result<_>>()?,
                },
            },
            typed_metadata: metadata
                .iter()
                .map(|item| {
                    Ok(ActionKind {
                        kind: item.require("type")?.string()?,
                    })
                })
                .collect::<Result<_>>()?,
        },
    };
    let content = CanonicalTable {
        action_contract: &table.action_contract,
        frame_hash: &table.frame_hash,
        kinds: &table.kinds,
        numerics: &table.numerics,
        offsets: &table.offsets,
        presence: &table.presence,
        purpose: &table.purpose,
        slots: &table.slots,
        synthetic_capacity: table.synthetic_capacity,
        targets: &table.targets,
    };
    check_columns(
        selected,
        frame,
        content,
        &table.table_hash,
        max_positions,
        output_bytes,
    )
}

fn check_columns<'a, Kind, Slot>(
    selected: usize,
    frame: Frame<'a>,
    table: CanonicalTable<'_, Kind, Slot>,
    table_hash: &'a str,
    max_positions: usize,
    output_bytes: usize,
) -> Result<Candidates<'a>>
where
    Kind: AsRef<str> + serde::Serialize,
    Slot: Copy + Into<usize> + serde::Serialize,
{
    ensure(
        (1..=128).contains(&max_positions),
        "invalid position capacity",
    )?;
    ensure(
        is_shared_card_instance_contract(table.action_contract)
            && table.purpose == "qualified_game"
            && !table.synthetic_capacity
            && frame.frame_hash == table.frame_hash
            && matches!(frame.actor, 1 | 2),
        "writer candidate frame or contract binding",
    )?;
    let n = table.kinds.len();
    let own = frame.payload.observation.own.cards;
    let elements = table.slots.len();
    ensure(
        (1..=65536).contains(&n)
            && table.targets.len() == n
            && table.numerics.len() == n
            && table.presence.len() == n
            && table.offsets.len() == n + 1
            && table.offsets[0] == 0
            && table.offsets[n] == elements
            && own.len() <= 18
            && selected < n,
        "writer candidate shape",
    )?;
    let permitted = frame
        .payload
        .typed_metadata
        .iter()
        .fold(0u16, |mask, item| {
            match KINDS.iter().position(|&kind| kind == item.kind) {
                Some(code) => mask | (1 << code),
                None => mask,
            }
        });
    let numeric_start = align_i64(2 * n);
    let presence_start = numeric_start + 32 * n;
    let offset_start = align_i64(presence_start + 4 * n);
    let slot_start = offset_start + 8 * (n + 1);
    // Existing shape/offset bounds imply at most 12 MiB, below codec2's 64 MiB.
    ensure(elements <= 18 * n, "invalid writer candidate element count")?;
    ensure(
        slot_start + elements == output_bytes,
        "writer candidate output size",
    )?;
    let mut binary = vec![0u8; output_bytes];
    for i in 0..n {
        let kind = table.kinds[i].as_ref();
        let code = KINDS
            .iter()
            .position(|&value| value == kind)
            .filter(|&code| permitted & (1 << code) != 0)
            .ok_or_else(|| error("writer candidate outside typed metadata"))?;
        let target = table.targets[i].as_ref();
        let targeted = matches!(kind, "play_sequence" | "sell" | "buy");
        ensure(
            if targeted {
                matches!(target, "self" | "opponent")
            } else {
                target == "omitted"
            },
            "writer candidate target presence",
        )?;
        let a = table.offsets[i];
        let b = table.offsets[i + 1];
        ensure(
            a <= b && b <= elements && b - a <= max_positions,
            "writer candidate row offsets",
        )?;
        let row = &table.slots[a..b];
        // The Python writer validates submission syntax, including arrays
        // that can be empty, independently of the generator's search rules.
        let cardinality_ok = match kind {
            "pray" | "buy_decision" => row.is_empty(),
            "sell" => row.len() == 2,
            "exchange" | "buy" | "invoke_phenomenon" => row.len() == 1,
            _ => true,
        };
        ensure(cardinality_ok, "writer candidate reference cardinality")?;
        let mut seen = 0u32;
        for (j, &slot) in row.iter().enumerate() {
            let slot = slot.into();
            ensure(slot < own.len(), "candidate slot outside original own view")?;
            let bit = 1 << slot;
            ensure(
                seen & bit == 0,
                "writer candidate repeats a physical reference",
            )?;
            seen |= bit;
            binary[slot_start + a + j] = slot as u8;
        }
        let presence = match kind {
            "exchange" => [true, true, true, false],
            "buy_decision" => [false, false, false, true],
            _ => [false; 4],
        };
        let values = table.numerics[i];
        ensure(
            table.presence[i] == presence
                && (!presence[0] && values[0] == 0
                    || presence[0] && i32::try_from(values[0]).is_ok())
                && (!presence[1] && values[1] == 0
                    || presence[1] && u32::try_from(values[1]).is_ok())
                && (!presence[2] && values[2] == 0
                    || presence[2] && u32::try_from(values[2]).is_ok())
                && (!presence[3] && values[3] == 0 || presence[3] && (0..=1).contains(&values[3])),
            "writer candidate numeric encoding",
        )?;
        binary[i] = code as u8;
        binary[n + i] = TARGETS.iter().position(|&value| value == target).unwrap() as u8;
        for j in 0..4 {
            let offset = numeric_start + (i * 4 + j) * 8;
            binary[offset..offset + 8].copy_from_slice(&values[j].to_le_bytes());
            binary[presence_start + 4 * i + j] = presence[j] as u8;
        }
        let offset = offset_start + 8 * i;
        binary[offset..offset + 8].copy_from_slice(&(a as i64).to_le_bytes());
    }
    binary[slot_start - 8..slot_start].copy_from_slice(&(elements as i64).to_le_bytes());
    let mut hasher = HashWriter(Sha256::new());
    serde_json::to_writer(&mut hasher, &table).map_err(|exc| error(exc.to_string()))?;
    ensure(
        format!("{:x}", hasher.0.finalize()) == table_hash,
        "writer candidate table hash",
    )?;
    let row = &table.slots[table.offsets[selected]..table.offsets[selected + 1]];
    let action = action_from_parts(
        table.kinds[selected].as_ref(),
        table.targets[selected].as_ref(),
        frame.actor,
        row.iter()
            .map(|&slot| Value::from(own[slot.into()].card_ref))
            .collect(),
        &table.numerics[selected],
        &table.presence[selected],
    )?;
    let action = serde_json::to_vec(&action).map_err(|exc| error(exc.to_string()))?;
    let sha256 = format!("{:x}", Sha256::digest(&binary));
    #[cfg(test)]
    let parsed_frame_bytes = vec_bytes(&own) + vec_bytes(&frame.payload.typed_metadata);
    Ok(Candidates {
        frame_hash: frame.frame_hash,
        table_hash,
        selected,
        count: n,
        elements,
        action,
        payload: binary,
        sha256,
        #[cfg(test)]
        parsed_columns_bytes: 0,
        #[cfg(test)]
        parsed_frame_bytes,
    })
}

#[cfg(test)]
fn vec_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity() * std::mem::size_of::<T>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn input(rows: usize, positions: usize) -> Value {
        let refs: Vec<_> = (0..18)
            .map(|i| json!({"card_ref":format!("c1_{i:032x}")}))
            .collect();
        let frame_hash = "a".repeat(64);
        let slots: Vec<_> = (0..rows).flat_map(|_| 0..positions).collect();
        let mut value = json!({
            "frame": {
                "actor": 2,
                "frame_hash":frame_hash,
                "payload":{
                    "observation":{"own":{"cards":refs}},
                    "typed_metadata":[{"type":"play_sequence"}],
                },
            },
            "generated":{"table":{
                "action_contract": shared_card_instance_contract(),
                "frame_hash": frame_hash,
                "kinds": vec!["play_sequence"; rows],
                "targets": vec!["opponent"; rows],
                "numerics": vec![[0; 4]; rows],
                "presence": vec![[false; 4]; rows],
                "offsets": (0..=rows).map(|i| i * positions).collect::<Vec<_>>(),
                "purpose": "qualified_game",
                "slots": slots,
                "synthetic_capacity": false,
            }},
            "selected": rows - 1,
        });
        sign(&mut value);
        value
    }

    fn sign(value: &mut Value) {
        let table = &mut value["generated"]["table"];
        table.as_object_mut().unwrap().remove("table_hash");
        let digest = Sha256::digest(serde_json::to_vec(table).unwrap());
        table["table_hash"] = json!(format!("{digest:x}"));
    }

    #[test]
    fn verifies_and_encodes_exact_codec_layout() {
        let capture = input(2, 3);
        let bytes = serde_json::to_vec(&capture).unwrap();
        let checked = check(&bytes, 18, 110).unwrap();
        assert_eq!(checked.selected, 1);
        assert_eq!((checked.count, checked.elements), (2, 6));
        assert_eq!(
            checked.table_hash,
            capture["generated"]["table"]["table_hash"]
        );
        assert_eq!(&checked.payload[..8], &[1, 1, 3, 3, 0, 0, 0, 0]);
        assert_eq!(checked.payload[8..80], [0; 72]);
        assert_eq!(&checked.payload[80..88], &0i64.to_le_bytes());
        assert_eq!(&checked.payload[88..96], &3i64.to_le_bytes());
        assert_eq!(&checked.payload[96..104], &6i64.to_le_bytes());
        assert_eq!(&checked.payload[104..], &[0, 1, 2, 0, 1, 2]);
        assert_eq!(
            checked.sha256,
            format!("{:x}", Sha256::digest(&checked.payload))
        );
        let action: Value = serde_json::from_slice(&checked.action).unwrap();
        assert_eq!(
            action,
            json!({"type":"play_sequence","target":1,
                   "ordered_card_refs": (0..3).map(|i| format!("c1_{i:032x}")).collect::<Vec<_>>()})
        );
        // Empty action arrays pass the writer's syntax validation even when
        // the search recipe could never generate a row with that shape.
        let empty = serde_json::to_vec(&input(1, 0)).unwrap();
        assert!(check(&empty, 18, 64).is_ok());
    }

    #[test]
    fn rejects_mutations_of_unselected_rows_shapes_and_hashes() {
        type Mutation = fn(&mut Value);
        let mutations: [(Mutation, &str); 7] = [
            (
                |v| v["generated"]["table"]["slots"][0] = json!(18),
                "outside original own view",
            ),
            (
                |v| v["generated"]["table"]["slots"][1] = json!(0),
                "reference",
            ),
            (
                |v| v["generated"]["table"]["targets"][0] = json!("null"),
                "target",
            ),
            (
                |v| v["generated"]["table"]["presence"][0][0] = json!(true),
                "numeric",
            ),
            (
                |v| v["generated"]["table"]["offsets"][1] = json!(7),
                "offsets",
            ),
            (
                |v| {
                    v["generated"]["table"]["action_contract"]["profile_id"] =
                        json!("card-instance-simulator-v2-20261104")
                },
                "contract",
            ),
            (|v| v["selected"] = json!(2), "shape"),
        ];
        for (mutate, failure) in mutations {
            let mut capture = input(2, 3);
            mutate(&mut capture);
            sign(&mut capture);
            let bytes = serde_json::to_vec(&capture).unwrap();
            let error = check(&bytes, 18, 110).err().unwrap();
            assert!(error.to_string().contains(failure), "{error}");
        }
        let mut capture = input(2, 3);
        capture["generated"]["table"]["table_hash"] = json!("a".repeat(64));
        let bytes = serde_json::to_vec(&capture).unwrap();
        assert!(
            check(&bytes, 18, 110)
                .err()
                .unwrap()
                .to_string()
                .contains("table hash")
        );
        assert!(
            check(&bytes, 18, 109)
                .err()
                .unwrap()
                .to_string()
                .contains("output size")
        );
    }

    #[test]
    fn rejects_valid_values_in_a_positional_table() {
        let mut capture = input(2, 3);
        let table = &capture["generated"]["table"];
        // Preserves the correctly hashed values, in their declaration order.
        let positional = [
            "action_contract",
            "frame_hash",
            "kinds",
            "numerics",
            "offsets",
            "presence",
            "purpose",
            "slots",
            "synthetic_capacity",
            "table_hash",
            "targets",
        ]
        .into_iter()
        .map(|name| table[name].clone())
        .collect();
        capture["generated"]["table"] = Value::Array(positional);
        let bytes = serde_json::to_vec(&capture).unwrap();
        assert!(
            check(&bytes, 18, 110)
                .err()
                .unwrap()
                .to_string()
                .contains("candidate table object")
        );
    }

    #[test]
    fn reports_capacity_for_representative_and_maximum_supported_columns() {
        // Recorder ownership differs before and after the call returns: the
        // parsed columns coexist with the native Vec, then PyBytes is copied
        // only after typed parse data have dropped.
        for (count, positions, expected_size) in [(1447, 18, 92622), (65536, 18, 4194312)] {
            let whole = serde_json::to_vec(&input(count, positions)).unwrap();
            let checked = check(&whole, 18, expected_size).unwrap();
            assert_eq!(checked.payload.capacity(), expected_size);
            eprintln!(
                "writer candidate capacities count={count} positions={positions} borrowed={}                  typed_columns={} typed_frame_vectors={} encoded_vec={} return_action_vec={}                  codec_allocation_overlap={}",
                whole.len(),
                checked.parsed_columns_bytes,
                checked.parsed_frame_bytes,
                checked.payload.capacity(),
                checked.action.capacity(),
                checked.payload.capacity() * 2,
            );
        }
    }
}
