//! Attested external HTTP v2. Management responses never enter policy input.
use super::{ensure, error, requests::wait_duration};
use crate::{
    budget::{Budget, Control},
    error::{Error, Result},
    json::{Value, ValueExt},
    structured_journal::card_instance,
};
use reqwest::{Client, Method};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Server {
    pub base_url: String,
    pub expected_fingerprint: String,
    pub expected_effective: Value,
    pub startup_info: Value,
    pub request_seconds: f64,
}
pub fn client() -> Result<Client> {
    Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .map_err(|e| Error::Io(e.to_string()))
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HttpFacts {
    pub responses: u64,
    pub error_responses: u64,
    pub transport_errors: u64,
}

pub struct Http {
    client: Client,
    server: Server,
    cached: Option<Value>,
    poisoned: bool,
    reconnect: bool,
    pub(crate) facts: HttpFacts,
    pub deadline: f64,
    pub metrics: Option<std::sync::Arc<super::metrics::Metrics>>,
}
impl Http {
    pub fn new(client: Client, server: Server, deadline: f64) -> Result<Self> {
        let url = reqwest::Url::parse(&server.base_url).map_err(|e| error(e.to_string()))?;
        ensure(
            url.scheme() == "http"
                && url.host_str() == Some("127.0.0.1")
                && url.port().is_some()
                && url.path() == "/"
                && url.query().is_none()
                && url.fragment().is_none()
                && url.username().is_empty()
                && url.password().is_none()
                && !server.base_url.ends_with('/'),
            "local managed server URL required",
        )?;
        ensure(
            server.request_seconds.is_finite() && server.request_seconds > 0.,
            "positive HTTP timeout required",
        )?;
        super::requests::checked_duration(server.request_seconds)?;
        Ok(Self {
            client,
            server,
            metrics: None,
            cached: None,
            poisoned: false,
            reconnect: false,
            facts: HttpFacts::default(),
            deadline,
        })
    }
    async fn wire(
        &mut self,
        method: Method,
        path: &str,
        body: Option<&[u8]>,
    ) -> std::result::Result<reqwest::Response, reqwest::Error> {
        let timeout = Duration::from_secs_f64(self.server.request_seconds)
            .min(wait_duration(self.deadline).unwrap_or(Duration::ZERO));
        let mut request = self
            .client
            .request(method, format!("{}{}", self.server.base_url, path))
            .timeout(timeout);
        if let Some(body) = body {
            request = request
                .header("content-type", "application/json")
                .body(body.to_vec());
        }
        let result = request.send().await;
        match &result {
            Ok(response) => {
                self.facts.responses += 1;
                if response.status().is_client_error() || response.status().is_server_error() {
                    self.facts.error_responses += 1;
                }
            }
            Err(_) => self.facts.transport_errors += 1,
        }
        result
    }
    pub async fn attest(&mut self) -> Result<()> {
        ensure(!self.poisoned, "attestation connection poisoned")?;
        let result = self.attest_inner().await;
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    async fn attest_inner(&mut self) -> Result<()> {
        if crate::budget::monotonic() >= self.deadline {
            return Err(Error::Deadline);
        }
        let response = self
            .wire(Method::GET, "/debug/v1/test-server/info", None)
            .await
            .map_err(|e| Error::Io(format!("attestation: {e}")))?;
        ensure(
            response.status() == 200
                && response
                    .headers()
                    .get("cache-control")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| {
                        v.split(',')
                            .any(|s| s.trim().eq_ignore_ascii_case("no-store"))
                    }),
            "attestation requires 200/no-store",
        )?;
        let bytes = response.bytes().await.map_err(|e| {
            self.facts.transport_errors += 1;
            error(e.to_string())
        })?;
        let value: Value = serde_json::from_slice(&bytes).map_err(|e| error(e.to_string()))?;
        value.closed(
            &[
                "schema_version",
                "server_kind",
                "server_instance_id",
                "seed_mode",
                "seed_fingerprint",
                "seed_use_limit",
                "max_invalid_attempts_per_decision",
                "waiting_for_players_timeout_secs",
                "player_action_timeout_secs",
                "max_game_turns",
                "server_idle_timeout_secs",
                "graceful_shutdown_timeout_secs",
            ],
            &[],
        )?;
        ensure(
            value["schema_version"].as_u64() == Some(1)
                && value["server_kind"] == "test_server"
                && value["seed_mode"] == "repeated"
                && value["seed_fingerprint"] == self.server.expected_fingerprint,
            "attestation seed/config mismatch",
        )?;
        let mut effective = value.clone();
        for key in [
            "server_kind",
            "server_instance_id",
            "seed_mode",
            "seed_fingerprint",
        ] {
            effective.as_object_mut().unwrap().remove(key);
        }
        ensure(
            effective == self.server.expected_effective
                && value == self.server.startup_info
                && self.cached.as_ref().is_none_or(|old| old == &value),
            "sticky server identity/effective config mismatch",
        )?;
        self.cached = Some(value);
        self.reconnect = false;
        Ok(())
    }
    pub async fn request(
        &mut self,
        method: Method,
        path: &str,
        payload: Option<&Value>,
        retry: bool,
    ) -> Result<Value> {
        ensure(
            self.cached.is_some() && !self.poisoned,
            "request forbidden before successful attestation",
        )?;
        if self.reconnect {
            self.attest().await?;
        }
        let body = payload
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|e| error(e.to_string()))?;
        for attempt in 0..=if retry { 2 } else { 0 } {
            if attempt > 0 {
                self.attest().await?;
            }
            if crate::budget::monotonic() >= self.deadline {
                return Err(Error::Deadline);
            }
            if path.contains("/events/") { if let Some(m) = &self.metrics { m.increment("event_transport_attempts", 1); } }
            let response = match self.wire(method.clone(), path, body.as_deref()).await {
                Ok(v) => v,
                Err(e) => {
                    self.reconnect = true;
                    if retry && attempt < 2 {
                        continue;
                    }
                    return Err(Error::Io(format!("delivery_unknown {method} {path}: {e}")));
                }
            };
            let status = response.status();
            if status == 204 {
                return Ok(Value::Null);
            }
            // A body-read failure is uncertain transport delivery too; retry the SAME command.
            let bytes = match response.bytes().await {
                Ok(v) => v,
                Err(e) => {
                    self.facts.transport_errors += 1;
                    self.reconnect = true;
                    if retry && attempt < 2 {
                        continue;
                    }
                    return Err(Error::Io(format!("delivery_unknown body: {e}")));
                }
            };
            if !status.is_success() {
                return Err(error(format!(
                    "simulator HTTP {} diagnostic {}",
                    status.as_u16(),
                    String::from_utf8_lossy(&bytes)
                )));
            }
            return serde_json::from_slice(&bytes)
                .map_err(|e| error(format!("invalid response JSON; no resend: {e}")));
        }
        unreachable!()
    }
    pub async fn create(&mut self) -> Result<String> {
        self.attest().await?;
        let info = self.request(Method::GET, "/v2/info", None, false).await?;
        info.closed(
            &[
                "name",
                "version",
                "api_version",
                "observation_schema_version",
                "action_schema_version",
                "action_options_schema_version",
                "public_event_schema_version",
                "card_ref_scheme",
                "rule_set_version",
                "random_algorithm",
                "random_algorithm_version",
                "card_catalog",
            ],
            &[],
        )?;
        for (key, pin) in [
            ("name", "gf-simulator"),
            ("api_version", "v2"),
            ("observation_schema_version", "2"),
            ("action_options_schema_version", "2"),
            ("public_event_schema_version", "2"),
            ("card_ref_scheme", "aes256-card-ref-v1"),
            ("rule_set_version", "true-match-1v1-v2"),
            ("random_algorithm", "xoshiro256pp"),
            ("random_algorithm_version", "1"),
        ] {
            ensure(info[key] == pin, "unsupported public info")?;
        }
        ensure(
            info["action_schema_version"].as_u64() == Some(2)
                && info["version"].as_str().is_some_and(|v| !v.is_empty()),
            "unsupported public info scalar",
        )?;
        ensure(
            info["card_catalog"]
                == json!({"catalog_schema_version":2,"catalog_version":"public-ja-v3","content_sha256":"f51a3c73feb8b3f01837e65768aafb5a3aa805a33483308130244b33b16a8ad6","definition_count":296,"locale":"ja","rule_set_version":"true-match-1v1-v2"}),
            "unsupported public catalog metadata",
        )?;
        let value=self.request(Method::POST,"/v2/games",Some(&json!({"rule_set_version":"true-match-1v1-v2","random_algorithm":"xoshiro256pp","random_algorithm_version":"1"})),false).await?;
        value.closed(
            &[
                "game_id",
                "status",
                "required_players",
                "joined_players",
                "state_revision",
                "rule_set_version",
                "random_algorithm",
                "random_algorithm_version",
            ],
            &[],
        )?;
        ensure(
            value["status"] == "waiting_for_players"
                && value["required_players"].as_u64() == Some(2)
                && value["joined_players"].as_u64() == Some(0)
                && value["state_revision"].is_null()
                && value["rule_set_version"] == "true-match-1v1-v2"
                && value["random_algorithm"] == "xoshiro256pp"
                && value["random_algorithm_version"] == "1",
            "invalid Create acknowledgement",
        )?;
        let game = value["game_id"].string()?.to_owned();
        ensure(
            game.len() == 26
                && game
                    .bytes()
                    .all(|b| b"0123456789ABCDEFGHJKMNPQRSTVWXYZ".contains(&b)),
            "invalid Simulator game ID",
        )?;
        // Caller owns the allocated game even if this fence fails; see Game::create.
        Ok(game)
    }
    pub async fn catalog(&mut self) -> Result<Value> {
        let catalog = self
            .request(Method::GET, "/v2/card-catalogs/public-ja-v3", None, false)
            .await?;
        let pins: Value = serde_json::from_slice(include_bytes!(
            "../../assets/card-instance-validation-pins-v2.json"
        ))
        .unwrap();
        let mut content = catalog.clone();
        content
            .as_object_mut()
            .ok_or_else(|| error("catalog object required"))?
            .remove("content_sha256");
        ensure(
            catalog["content_sha256"] == pins["action_contract"]["catalog_sha256"]
                && super::hash(&content)? == pins["action_contract"]["catalog_sha256"],
            "public catalog hash mismatch",
        )?;
        let definitions = catalog["definitions"]
            .as_array()
            .ok_or_else(|| error("catalog definitions required"))?;
        Ok(Value::Object(
            definitions
                .iter()
                .map(|v| Ok((v["definition_id"].string()?.to_owned(), v.clone())))
                .collect::<Result<_>>()?,
        ))
    }
    pub async fn join(&mut self, game: &str, command: &str, viewer: u8) -> Result<()> {
        let v = self
            .request(
                Method::POST,
                &format!("/v2/games/{game}/players"),
                Some(&json!({"command_id":command})),
                true,
            )
            .await?;
        v.closed(
            &[
                "game_id",
                "player_id",
                "status",
                "joined_players",
                "required_players",
                "state_revision",
            ],
            &[],
        )?;
        ensure(
            v["game_id"] == game
                && v["player_id"] == viewer
                && v["joined_players"] == viewer
                && v["required_players"] == 2
                && v["status"]
                    == if viewer == 1 {
                        "waiting_for_players"
                    } else {
                        "active"
                    }
                && (viewer == 1) == v["state_revision"].is_null(),
            "invalid Join acknowledgement",
        )?;
        if !v["state_revision"].is_null() {
            v["state_revision"].integer(0, u64::MAX as i128)?;
        }
        Ok(())
    }
    pub async fn observe(&mut self, game: &str, viewer: u8) -> Result<Value> {
        let v = self
            .request(
                Method::GET,
                &format!("/v2/games/{game}/observations/{viewer}"),
                None,
                false,
            )
            .await?;
        card_instance::observation(
            &v,
            Some(viewer as i128),
            &Budget::new(12usize << 30),
            &Control::new(f64::MAX)?,
        )?;
        Ok(v)
    }
    pub async fn events(
        &mut self,
        game: &str,
        viewer: u8,
        mut cursor: Option<String>,
        max_events: usize,
    ) -> Result<(Vec<Value>, Option<String>)> {
        let mut events = vec![];
        loop {
            let mut url = reqwest::Url::parse(&format!(
                "{}/v2/games/{game}/events/{viewer}",
                self.server.base_url
            ))
            .map_err(|e| error(e.to_string()))?;
            url.query_pairs_mut().append_pair("limit", "1024");
            if let Some(c) = &cursor {
                url.query_pairs_mut().append_pair("after", c);
            }
            let path = format!("{}?{}", url.path(), url.query().unwrap());
            let metrics = self.metrics.clone();
            let mut span = metrics.as_ref().map(|m| super::metrics::PageSpan::new(m));
            let page = self.request(Method::GET, &path, None, false).await?;
            page.closed(
                &[
                    "public_event_schema_version",
                    "rule_set_version",
                    "events",
                    "next_cursor",
                    "has_more",
                ],
                &[],
            )?;
            ensure(
                page["public_event_schema_version"] == "2"
                    && page["rule_set_version"] == "true-match-1v1-v2",
                "unsupported event feed",
            )?;
            let rows = page["events"].array()?;
            ensure(rows.len() <= 1024, "event page exceeds requested limit")?;
            ensure(
                events.len() + rows.len() <= max_events,
                "configured interval/episode event capacity exceeded",
            )?;
            let next = if page["next_cursor"].is_null() {
                None
            } else {
                Some(page["next_cursor"].string()?.to_owned())
            };
            let more = page["has_more"].boolean()?;
            ensure(
                !more || next.is_some() && next != cursor,
                "event pagination made no progress",
            )?;
            if let Some(span) = &mut span { span.success(); }
            drop(span);
            events.extend(rows.iter().cloned());
            cursor = next;
            if !more {
                return Ok((events, cursor));
            }
        }
    }
    pub async fn options(&mut self, game: &str, viewer: u8) -> Result<Value> {
        self.request(
            Method::GET,
            &format!("/v2/games/{game}/action-options/{viewer}"),
            None,
            false,
        )
        .await
    }
    pub async fn act(&mut self, game: &str, viewer: u8, pending: &Value) -> Result<Value> {
        let v = self
            .request(
                Method::POST,
                &format!("/v2/games/{game}/players/{viewer}/actions"),
                Some(pending),
                true,
            )
            .await?;
        Self::validate_act_ack(&v, viewer, pending)?;
        // Validation only: ACK events/options/next_observation never enter history.
        Ok(v)
    }
    fn validate_act_ack(v: &Value, viewer: u8, pending: &Value) -> Result<()> {
        v.closed(
            &[
                "public_event_schema_version",
                "rule_set_version",
                "card_ref_scheme",
                "command_id",
                "previous_revision",
                "state_revision",
                "status",
                "events",
                "result",
                "next_observation",
                "action_options",
            ],
            &[],
        )?;
        ensure(
            v["public_event_schema_version"] == "2"
                && v["rule_set_version"] == "true-match-1v1-v2"
                && v["card_ref_scheme"] == "aes256-card-ref-v1",
            "unsupported Act acknowledgement",
        )?;
        ensure(
            v["command_id"] == pending["command_id"]
                && v["previous_revision"] == pending["expected_revision"]
                && v["state_revision"]
                    .as_u64()
                    .is_some_and(|r| r > pending["expected_revision"].as_u64().unwrap()),
            "invalid action acknowledgement identity",
        )?;
        ensure(
            ["active", "terminal", "aborted"]
                .iter()
                .any(|s| v["status"] == *s),
            "invalid ACK status",
        )?;
        crate::validation::result(&v["result"])?;
        ensure(
            (v["status"] == "terminal") == !v["result"].is_null(),
            "ACK result/status mismatch",
        )?;
        let budget = Budget::new(12usize << 30);
        let control = Control::new(f64::MAX)?;
        let recipe = crate::structured_tokens::Recipe::corrected(16, &budget, &control)?;
        for event in v["events"].array()? {
            card_instance::event(event, &recipe, &budget, &control)?;
        }
        let obs = &v["next_observation"];
        if !obs.is_null() {
            card_instance::observation_context(
                obs,
                Some(viewer as i128),
                &budget,
                &control,
                v["status"] != "active",
            )?;
            ensure(
                obs["state_revision"] == v["state_revision"] && obs["result"] == v["result"],
                "ACK observation binding mismatch",
            )?;
        }
        if !v["action_options"].is_null() {
            ensure(
                !obs.is_null() && v["status"] == "active" && obs["actionable"] == true,
                "ACK hints require active own observation",
            )?;
            card_instance::hints(
                &json!({"state_revision":v["state_revision"],"viewer_player_id":viewer,"action_options":v["action_options"]}),
                obs,
                &control,
            )?;
        }
        Ok(())
    }
    pub async fn replay(&mut self, game: &str, viewer: u8) -> Result<Value> {
        self.request(
            Method::GET,
            &format!("/v2/games/{game}/replay/{viewer}"),
            None,
            false,
        )
        .await
    }
    pub async fn delete(&mut self, game: &str) -> Result<()> {
        self.request(Method::DELETE, &format!("/v2/games/{game}"), None, false)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_ack_accepts_only_the_draw_form_of_the_concealed_marker() {
        let pending = json!({"command_id":"5d5f7f78-30ec-4a23-b39b-33fc4a378b84",
                             "expected_revision":6});
        let marker = json!({"type":"artifact_drawn","player":1,"definition_id":"concealed"});
        let mut ack = json!({
            "public_event_schema_version":"2",
            "rule_set_version":"true-match-1v1-v2",
            "card_ref_scheme":"aes256-card-ref-v1",
            "command_id":pending["command_id"],
            "previous_revision":6,
            "state_revision":7,
            "status":"active",
            "events":vec![marker; 8],
            "result":null,
            "next_observation":null,
            "action_options":null
        });
        Http::validate_act_ack(&ack, 1, &pending).unwrap();
        assert_eq!(ack["events"].array().unwrap().len(), 8);
        ack["events"][0]["type"] = json!("artifact_used");
        assert!(Http::validate_act_ack(&ack, 1, &pending).is_err());
    }
}
