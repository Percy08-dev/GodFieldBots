//! Charged open-addressed indices for owned parser and file-evidence containers.
use crate::{
    budget::{Budget, BudgetVec, Control},
    error::{Error, Result},
};
use std::{
    hash::{DefaultHasher, Hasher},
    sync::Arc,
};

#[derive(Clone, Copy)]
struct Slot {
    hash: u64,
    index: usize,
    occupied: bool,
}
const EMPTY: Slot = Slot {
    hash: 0,
    index: 0,
    occupied: false,
};
pub struct Index {
    slots: BudgetVec<Slot>,
    length: usize,
}
impl Index {
    pub fn new(budget: &Arc<Budget>) -> Self {
        Self {
            slots: BudgetVec::new(budget),
            length: 0,
        }
    }
    pub fn find(
        &self,
        hash: u64,
        matches: impl Fn(usize) -> bool,
        control: &Control,
    ) -> Result<Option<usize>> {
        if self.slots.is_empty() {
            return Ok(None);
        }
        let mut position = hash as usize & (self.slots.len() - 1);
        for _ in 0..self.slots.len() {
            control.check()?;
            let slot = self.slots[position];
            if !slot.occupied {
                return Ok(None);
            }
            if slot.hash == hash && matches(slot.index) {
                return Ok(Some(slot.index));
            }
            position = (position + 1) & (self.slots.len() - 1);
        }
        Err(Error::contract("native lookup index is full"))
    }
    pub fn insert(&mut self, hash: u64, index: usize, control: &Control) -> Result<()> {
        if self.length + 1 > self.slots.len() / 2 {
            let count = self
                .slots
                .len()
                .checked_mul(2)
                .ok_or_else(|| Error::resource("lookup capacity overflow"))?
                .max(8);
            let mut replacement = BudgetVec::with_capacity(self.slots.budget(), count)?;
            for _ in 0..count {
                replacement.push(EMPTY)?;
            }
            for slot in self.slots.iter().filter(|slot| slot.occupied) {
                control.check()?;
                insert_slot(&mut replacement, *slot, control)?;
            }
            self.slots = replacement;
        }
        insert_slot(
            &mut self.slots,
            Slot {
                hash,
                index,
                occupied: true,
            },
            control,
        )?;
        self.length += 1;
        Ok(())
    }
}
fn insert_slot(slots: &mut [Slot], slot: Slot, control: &Control) -> Result<()> {
    let mut position = slot.hash as usize & (slots.len() - 1);
    loop {
        control.check()?;
        if !slots[position].occupied {
            slots[position] = slot;
            return Ok(());
        }
        position = (position + 1) & (slots.len() - 1);
    }
}
pub fn hash(bytes: &[u8], control: &Control) -> Result<u64> {
    let mut hasher = DefaultHasher::new();
    for block in bytes.chunks(4096) {
        control.check()?;
        hasher.write(block);
    }
    Ok(hasher.finish())
}
